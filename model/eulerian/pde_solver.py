"""
pde_solver.py -- Direct Eulerian optimal-control solver for the mixing problem
on Omega = (0,1)^2.

Problem (Sec 2 of tex/main.tex)
-------------------------------
    min_v   int_0^T int_Omega rho ||v||^2 dx dt
          + ||rho(T) - mean(rho0)||^2_{H^{-1}}
    s.t.    d_t rho + div(v rho) = 0,  rho(0) = rho_0,
            div(v) = 0 in Omega,  v . n = 0 on dOmega.

Design notes
------------
- v is parameterized as a time-dependent combination of the SAME div-free
  cellular-flow basis used by the neural network in ../basis.py:

      e_{mn}(x) = c_{mn} * ( n*pi sin(m*pi*x1) cos(n*pi*x2),
                            -m*pi cos(m*pi*x1) sin(n*pi*x2) )
      c_{mn}    = 2 / (pi * sqrt(m^2 + n^2))

  By construction, div(e_{mn}) = 0 exactly and e_{mn} . n = 0 on dOmega,
  so incompressibility + no-flow BC hold structurally; no Hodge projection
  is needed. The unknown is lam_k in R^L at the N_T time midpoints
  (L = M_basis ** 2).

- rho is discretised on an N_x x N_x cell-centred MAC grid; v1 and v2 live on
  cell faces. Advection uses a second-order MUSCL (minmod) upwind flux + RK2
  (Heun) in time. Since v is analytically div-free, mass is conserved up to
  rounding.

- Running cost reduces exactly to  dt * sum_k ||lam_k||_2^2  because:
    (a) the basis is L^2-orthonormal,
    (b) for structurally div-free flows and uniform rho0 (the case we care
        about), rho is transported as a passive scalar and integral_Omega rho
        is invariant, but more importantly the *kinetic form* we target in
        the Lagrangian code is ||v||^2_{L^2}, which coincides with the
        transported form up to a constant for indicator rho0. We use the
        un-weighted ||v||^2_{L^2} version (same as train.py) for direct
        comparability with the particle solver.
  (Proved in Prop 3.1 of tex/main.tex.)

- Mix-norm on the grid uses DCT-II (Neumann cosine basis, consistent with
  basis.SpectralMixingCost) via torch.fft on the even-reflected field.

Optimisation: torch.optim.LBFGS with strong-Wolfe, zero init.
"""

from __future__ import annotations

import os
import math
import argparse

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cellular-flow basis on the MAC staggered grid
# ---------------------------------------------------------------------------

def _basis_on_faces(M_basis: int, N_x: int, device, dtype=torch.float64):
    """
    Pre-compute e_{mn} evaluated on the two MAC face grids.

    Returns two tensors
        E1 : (L, N_x + 1, N_x)   basis values at u-faces
        E2 : (L, N_x, N_x + 1)   basis values at v-faces
    with  L = M_basis ** 2  and modes iterated (m, n) with m, n = 1..M_basis
    (same order as basis.DivFreeBasis).

    Face-coordinate layout
      u-face (i, j+1/2) : x1 = i / N_x,        x2 = (j + 0.5) / N_x
      v-face (i+1/2, j) : x1 = (i + 0.5) / N_x, x2 = j / N_x
    """
    pi = math.pi
    h  = 1.0 / N_x

    # u-face coordinates
    x1_u = torch.arange(N_x + 1, device=device, dtype=dtype) * h        # (N_x+1,)
    x2_u = (torch.arange(N_x, device=device, dtype=dtype) + 0.5) * h    # (N_x,)

    # v-face coordinates
    x1_v = (torch.arange(N_x, device=device, dtype=dtype) + 0.5) * h    # (N_x,)
    x2_v = torch.arange(N_x + 1, device=device, dtype=dtype) * h        # (N_x+1,)

    E1_list, E2_list = [], []
    for m in range(1, M_basis + 1):
        for n in range(1, M_basis + 1):
            c_mn = 2.0 / (pi * math.sqrt(m * m + n * n))
            # u-component: n*pi * sin(m pi x1) * cos(n pi x2)
            sin_mx1_u = torch.sin(m * pi * x1_u)            # (N_x+1,)
            cos_nx2_u = torch.cos(n * pi * x2_u)            # (N_x,)
            e1 = (n * pi * c_mn) * sin_mx1_u[:, None] * cos_nx2_u[None, :]   # (N_x+1, N_x)

            # v-component: -m*pi * cos(m pi x1) * sin(n pi x2)
            cos_mx1_v = torch.cos(m * pi * x1_v)            # (N_x,)
            sin_nx2_v = torch.sin(n * pi * x2_v)            # (N_x+1,)
            e2 = -(m * pi * c_mn) * cos_mx1_v[:, None] * sin_nx2_v[None, :]  # (N_x, N_x+1)

            E1_list.append(e1)
            E2_list.append(e2)

    E1 = torch.stack(E1_list, dim=0)   # (L, N_x+1, N_x)
    E2 = torch.stack(E2_list, dim=0)   # (L, N_x, N_x+1)

    # Enforce exact v.n = 0 on dOmega (redundant since sin(m pi 0)=sin(m pi 1)=0
    # analytically, but guards against floating-point leakage).
    E1[:, 0, :]  = 0.0
    E1[:, -1, :] = 0.0
    E2[:, :, 0]  = 0.0
    E2[:, :, -1] = 0.0

    return E1, E2


def _assemble_velocity(lam_k: torch.Tensor,
                       E1: torch.Tensor,
                       E2: torch.Tensor):
    """
    lam_k : (L,)
    Returns v1 : (N_x+1, N_x), v2 : (N_x, N_x+1)
    """
    v1 = torch.einsum('l,lij->ij', lam_k, E1)
    v2 = torch.einsum('l,lij->ij', lam_k, E2)
    return v1, v2


# ---------------------------------------------------------------------------
# MUSCL (minmod) upwind transport
# ---------------------------------------------------------------------------

def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Componentwise minmod limiter: 0 if different signs, else signed min|.|."""
    same_sign = (a * b) > 0.0
    mag = torch.minimum(a.abs(), b.abs())
    return torch.where(same_sign, torch.sign(a) * mag, torch.zeros_like(a))


def _advect_rhs(rho: torch.Tensor,
                v1: torch.Tensor,
                v2: torch.Tensor,
                h: float) -> torch.Tensor:
    """
    Compute -(dF1/dx + dF2/dy)/h   (shape (N_x, N_x), same as rho).

    Fluxes on internal faces use a second-order MUSCL (minmod) upwind
    reconstruction of rho. No-flow boundaries are enforced by taking v.n = 0
    (which pde_solver guarantees by construction of the basis) — so the
    contributions at boundary faces are zero and we simply skip them.

    Layout:
      rho        : (N_x, N_x)     cell centres, rho[i, j] = rho at
                                   (i + 0.5, j + 0.5) h
      v1 (u)     : (N_x+1, N_x)   u[i, j]   = v1 at (i, j + 0.5) h
      v2 (w)     : (N_x, N_x+1)   w[i, j]   = v2 at (i + 0.5, j) h
    """
    Nx = rho.shape[0]

    # ---- x-direction (u-faces) ---------------------------------------------
    # Interior u-faces are i = 1 .. Nx-1 (boundary faces have v.n = 0).
    # At face i, between cells (i-1, j) and (i, j).
    # Cell-centre slopes (minmod):  dx_rho[i, j] = minmod(rho[i+1]-rho[i],
    #                                                      rho[i]-rho[i-1])
    # Need slopes for i = 0 .. Nx-1; we compute them with zero slopes at edges.
    rho_pad_x = F.pad(rho, (0, 0, 1, 1))          # (Nx+2, Nx): pad i
    diff_fwd  = rho_pad_x[2:, :]  - rho_pad_x[1:-1, :]    # rho[i+1] - rho[i]
    diff_bwd  = rho_pad_x[1:-1, :] - rho_pad_x[:-2, :]    # rho[i]   - rho[i-1]
    slope_x   = _minmod(diff_fwd, diff_bwd)                # (Nx, Nx)
    # Left and right reconstructions at each cell's x-faces.
    rho_L = rho - 0.5 * slope_x       # value at face i, from the LEFT cell
    rho_R = rho + 0.5 * slope_x       # value at face i+1, from the RIGHT cell
                                      # (i.e. rho_R[i-1,j] approaches face i
                                      #  from its right-edge reconstruction)

    # For internal u-face i (1..Nx-1): left cell is (i-1), right cell is (i).
    u_int = v1[1:-1, :]                              # (Nx-1, Nx)
    face_from_left  = rho_R[:-1, :]                  # rho_R[i-1, j]
    face_from_right = rho_L[1:,  :]                  # rho_L[i,   j]
    rho_face_x = torch.where(u_int >= 0.0, face_from_left, face_from_right)
    F1_int = u_int * rho_face_x                      # (Nx-1, Nx)

    # Full u-flux with zero contribution at domain boundaries.
    F1 = rho.new_zeros(Nx + 1, Nx)
    F1[1:-1, :] = F1_int
    # (F1[0,:] = F1[-1,:] = 0 since v1 = 0 there.)

    # ---- y-direction (v-faces) ---------------------------------------------
    rho_pad_y = F.pad(rho, (1, 1, 0, 0))          # (Nx, Nx+2): pad j
    diff_fwd_y = rho_pad_y[:, 2:]  - rho_pad_y[:, 1:-1]
    diff_bwd_y = rho_pad_y[:, 1:-1] - rho_pad_y[:, :-2]
    slope_y    = _minmod(diff_fwd_y, diff_bwd_y)           # (Nx, Nx)
    rho_B = rho - 0.5 * slope_y
    rho_T = rho + 0.5 * slope_y

    w_int  = v2[:, 1:-1]                              # (Nx, Nx-1)
    face_from_below = rho_T[:, :-1]
    face_from_above = rho_B[:,  1:]
    rho_face_y = torch.where(w_int >= 0.0, face_from_below, face_from_above)
    F2_int = w_int * rho_face_y

    F2 = rho.new_zeros(Nx, Nx + 1)
    F2[:, 1:-1] = F2_int

    # Divergence of the flux:  (F1[i+1,j] - F1[i,j])/h + (F2[i,j+1] - F2[i,j])/h
    div = (F1[1:, :] - F1[:-1, :] + F2[:, 1:] - F2[:, :-1]) / h
    return -div


def advect_step(rho: torch.Tensor,
                v1: torch.Tensor, v2: torch.Tensor,
                dt: float, h: float) -> torch.Tensor:
    """One Heun (RK2) step of the finite-volume transport."""
    rhs1  = _advect_rhs(rho,                 v1, v2, h)
    rho_1 = rho + dt * rhs1
    rhs2  = _advect_rhs(rho_1,               v1, v2, h)
    return rho + 0.5 * dt * (rhs1 + rhs2)


def _max_abs_velocity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    return torch.maximum(v1.abs().max(), v2.abs().max())


def advect_interval(rho: torch.Tensor,
                    v1: torch.Tensor, v2: torch.Tensor,
                    dt: float, h: float,
                    cfl: float = 0.5) -> torch.Tensor:
    """Advance rho by dt under (v1, v2) with CFL-limited substepping."""
    vmax = _max_abs_velocity(v1, v2)
    if vmax.item() * dt / h <= cfl or vmax.item() == 0.0:
        return advect_step(rho, v1, v2, dt, h)

    n_sub = int(math.ceil(vmax.item() * dt / (cfl * h)))
    sub_dt = dt / n_sub
    out = rho
    for _ in range(n_sub):
        out = advect_step(out, v1, v2, sub_dt, h)
    return out


# ---------------------------------------------------------------------------
# Grid H^{-1} mix-norm via DCT-II (Neumann cosine basis)
# ---------------------------------------------------------------------------

def _dct2_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Type-II DCT along the last two axes, matching scipy.fft.dctn(norm=None).
    Implemented via the standard even-reflection trick on torch.fft.rfft.

    Input  : (..., N, N)
    Output : (..., N, N)  Y[..., k1, k2] = sum_{i, j} 4 * cos(pi k1 (i+0.5)/N)
                                                      * cos(pi k2 (j+0.5)/N)
                                                      * x[..., i, j]
    (i.e. the 'unnormalised' DCT-II convention.)
    """
    # 1-D DCT-II: even-reflect along the axis, then take the real part of an
    # N-length rfft of the reflected sequence, with phase twist.
    def _dct1(u):
        N = u.shape[-1]
        y = torch.cat([u, u.flip(-1)], dim=-1)           # length 2N
        Y = torch.fft.rfft(y, n=2 * N, dim=-1)           # (..., N+1)
        k = torch.arange(N + 1, device=u.device, dtype=u.dtype)
        phase = torch.exp(-1j * math.pi * k / (2 * N))
        out = (Y * phase).real[..., :N]
        return out

    return _dct1(_dct1(x).transpose(-1, -2)).transpose(-1, -2)


def grid_mix_norm_sq(rho: torch.Tensor,
                     K_max: int | None = None,
                     dye: bool = False) -> torch.Tensor:
    """
    Discrete ||rho - mean(rho)||^2_{H^{-1}(Omega)} on an N_x x N_x cell-centred
    grid over Omega = (0,1)^2.

    Parameters
    ----------
    K_max : int, optional
        If given, truncate the spectral sum at k1, k2 <= K_max (both axes).
        This matches basis.SpectralMixingCost / DyeMixCost with the same
        K_max.  If None, use the full grid spectrum.
    dye : bool
        If True, compute the dye mix-norm ||2 rho - 1||^2_{H^{-1}} instead
        of the plain density mix-norm.  For a half-domain initial condition
        this equals the DyeMixCost used in train.py (since f = rho_1 - rho_2
        = 2 rho_1 - 1 when rho_1 + rho_2 = 1), so `dye=True, K_max=12` makes
        the Eulerian loss numerically identical to the network's training
        loss (up to discretisation of rho).

    Consistent with basis.SpectralMixingCost:
      rho_hat_k = int_Omega rho(x) * a_{k1} a_{k2} cos(k1 pi x1) cos(k2 pi x2) dx
      ||rho - mean||^2_{H^-1} = sum_{k != 0} rho_hat_k^2 / (pi^2 (k1^2 + k2^2))

    Midpoint quadrature:
      rho_hat_k = a_{k1} a_{k2} * int rho cos cos dx
                ~ a_{k1} a_{k2} * h^2 * sum_{i,j} rho[i,j] cos(...) cos(...).
    The scipy-style DCT-II _dct1 above carries a factor 2 per axis (norm=None
    convention), so the 2-D DCT-II returns 4 * sum(rho cos cos); we therefore
    divide by 4 to recover the quadrature sum.
    """
    Nx = rho.shape[-1]
    h  = 1.0 / Nx
    coeff = _dct2_2d(rho.to(torch.float64))           # (Nx, Nx)

    device, dtype = rho.device, torch.float64
    k = torch.arange(Nx, device=device, dtype=dtype)
    a = torch.where(k == 0,
                    torch.ones_like(k),
                    math.sqrt(2.0) * torch.ones_like(k))
    ak1k2 = a[:, None] * a[None, :]                    # (Nx, Nx)

    rho_hat = (h * h / 4.0) * ak1k2 * coeff            # (Nx, Nx)

    pi = math.pi
    eigs = pi * pi * (k[:, None] ** 2 + k[None, :] ** 2)
    eigs[0, 0] = 1.0  # dummy; zero it out in the sum

    weight = 1.0 / eigs
    weight[0, 0] = 0.0
    if K_max is not None:
        mask = (k[:, None] <= K_max) & (k[None, :] <= K_max)
        weight = weight * mask.to(weight.dtype)

    out = (rho_hat * rho_hat * weight).sum().to(rho.dtype)
    if dye:
        # ||2 rho - 1||^2_{H^-1} = 4 * ||rho - mean||^2_{H^-1}  for k != 0
        # (the constant '1' only contributes to the k=0 mode, which is
        # excluded from the H^-1 sum).
        out = 4.0 * out
    return out


# ---------------------------------------------------------------------------
# Half-domain initial density
# ---------------------------------------------------------------------------

def half_domain_rho0(N_x: int, device, dtype=torch.float64) -> torch.Tensor:
    """rho_0(x) = 1  if x1 < 0.5 else 0   on the cell-centred grid."""
    h  = 1.0 / N_x
    x1 = (torch.arange(N_x, device=device, dtype=dtype) + 0.5) * h
    out = torch.zeros(N_x, N_x, device=device, dtype=dtype)
    out[x1 < 0.5, :] = 1.0
    return out


# ---------------------------------------------------------------------------
# Forward simulation (no autograd on the control params needed by the caller)
# ---------------------------------------------------------------------------

def simulate(lam: torch.Tensor,
             rho0: torch.Tensor,
             E1: torch.Tensor,
             E2: torch.Tensor,
             T: float,
             cfl: float = 0.5,
             record_rho: bool = False):
    """
    Integrate rho under the time-dependent basis control lam (N_T, L).

    Returns
    -------
    rho_T : (N_x, N_x)
    traj  : (N_T + 1, N_x, N_x)  or  None
    """
    N_T = lam.shape[0]
    Nx  = rho0.shape[0]
    h   = 1.0 / Nx
    dt  = T / N_T

    rho = rho0
    if record_rho:
        traj = [rho.detach().clone()]
    for k in range(N_T):
        v1, v2 = _assemble_velocity(lam[k], E1, E2)
        rho = advect_interval(rho, v1, v2, dt, h, cfl=cfl)
        if record_rho:
            traj.append(rho.detach().clone())
    if record_rho:
        return rho, torch.stack(traj, dim=0)
    return rho, None


# ---------------------------------------------------------------------------
# Loss and optimisation
# ---------------------------------------------------------------------------

def compute_loss(lam: torch.Tensor,
                 rho0: torch.Tensor,
                 E1: torch.Tensor,
                 E2: torch.Tensor,
                 T: float,
                 lam_mix: float,
                 lam_kin: float,
                 K_mix: int | None = 12,
                 dye: bool = True,
                 cfl: float = 0.5):
    """
    Differentiable loss in lam.  Defaults match train.py's DyeMixCost /
    Lagrangian kinetic: K_mix=12, dye=True, lam_kin=1e-3.

    E_kin = dt * sum_k ||lam_k||_2^2           (orthonormal basis; matches the
                                                particle kinetic when rho is a
                                                probability density)
    E_mix = grid_mix_norm_sq(rho_T, K_max=K_mix, dye=dye)
    """
    N_T = lam.shape[0]
    dt  = T / N_T

    rho_T, _ = simulate(lam, rho0, E1, E2, T, cfl=cfl, record_rho=False)

    E_kin = dt * (lam ** 2).sum()
    E_mix = grid_mix_norm_sq(rho_T, K_max=K_mix, dye=dye)

    loss = lam_mix * E_mix + lam_kin * E_kin
    return loss, E_mix, E_kin, rho_T


def solve_eulerian(rho0: torch.Tensor,
                   M_basis: int = 2,
                   N_T: int = 20,
                   T: float = 5.0,
                   lam_mix: float = 1.0,
                   lam_kin: float = 1e-3,
                   K_mix: int | None = 12,
                   dye: bool = True,
                   iters: int = 200,
                   cfl: float = 0.5,
                   lbfgs_lr: float = 1.0,
                   tol: float = 1e-6,
                   device: str | None = None,
                   init_noise: float = 1e-3,
                   seed: int = 0,
                   verbose: bool = True) -> dict:
    """
    Solve the Eulerian optimal-control problem on the fixed N_x x N_x grid
    given by `rho0.shape`.

    Parameters
    ----------
    rho0    : (N_x, N_x)       cell-centred initial density
    M_basis : int              div-free basis wavenumber (L = M_basis^2)
    N_T     : int              number of control time intervals
    T       : float            time horizon
    lam_mix, lam_kin : float   weights in the loss
    iters   : int              max L-BFGS iterations
    cfl     : float            transport CFL target (substepping if violated)

    Returns
    -------
    dict(lam           : (N_T, L),
         rho_traj      : (N_T+1, N_x, N_x),
         E_mix_curve   : (N_T+1,),    dot-H^-1 squared along the trajectory
         E_kin         : float,
         loss_history  : list[float])
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rho0 = rho0.to(device=device, dtype=torch.float64)
    Nx   = rho0.shape[0]
    L    = M_basis ** 2

    E1, E2 = _basis_on_faces(M_basis, Nx, device=device, dtype=torch.float64)

    # Small random init: exactly lam = 0 is a degenerate stationary point of
    # the discrete upwind loss (the sign branch in torch.where gives zero
    # gradient through F when v = 0). A tiny perturbation breaks the tie.
    gen = torch.Generator(device='cpu').manual_seed(seed)
    lam0 = init_noise * torch.randn(N_T, L, generator=gen).to(
        device=device, dtype=torch.float64)
    lam = lam0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS([lam], lr=lbfgs_lr,
                                  max_iter=iters,
                                  tolerance_grad=1e-10,
                                  tolerance_change=tol,
                                  history_size=50,
                                  line_search_fn='strong_wolfe')

    loss_history: list[float] = []

    def closure():
        optimizer.zero_grad()
        loss, E_mix, E_kin, _ = compute_loss(
            lam, rho0, E1, E2, T, lam_mix, lam_kin,
            K_mix=K_mix, dye=dye, cfl=cfl)
        loss.backward()
        loss_history.append(loss.item())
        if verbose and (len(loss_history) % 10 == 1 or len(loss_history) <= 5):
            print(f"  [LBFGS {len(loss_history):3d}] "
                  f"loss={loss.item():.6e}  "
                  f"E_mix={E_mix.item():.6e}  "
                  f"E_kin={E_kin.item():.6e}")
        return loss

    optimizer.step(closure)

    # Final trajectory + mix-norm curve with optimised lam
    with torch.no_grad():
        rho_T, traj = simulate(lam.detach(), rho0, E1, E2, T,
                               cfl=cfl, record_rho=True)
        E_kin_final = (T / N_T) * (lam.detach() ** 2).sum().item()
        E_mix_curve = torch.stack([
            grid_mix_norm_sq(traj[k], K_max=K_mix, dye=dye)
            for k in range(traj.shape[0])
        ]).cpu().numpy()

    return dict(
        lam          = lam.detach().cpu(),
        rho_traj     = traj.cpu(),
        rho_T        = rho_T.cpu(),
        E_mix_curve  = E_mix_curve,
        E_kin        = E_kin_final,
        loss_history = loss_history,
        meta = dict(M_basis=M_basis, N_T=N_T, N_x=Nx, T=T,
                    lam_mix=lam_mix, lam_kin=lam_kin,
                    K_mix=K_mix, dye=dye, iters=iters, cfl=cfl),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Eulerian PDE solver for mixing")
    p.add_argument('--M_basis', type=int,   default=2)
    p.add_argument('--K_mix',   type=int,   default=12,
                   help='spectral truncation (default matches train.py)')
    p.add_argument('--N_x',     type=int,   default=128)
    p.add_argument('--N_T',     type=int,   default=20,
                   help='control/ODE time-step count (default matches train.py)')
    p.add_argument('--T',       type=float, default=5.0)
    p.add_argument('--lam_mix', type=float, default=1.0)
    p.add_argument('--lam_kin', type=float, default=1e-3,
                   help='default matches train.py')
    p.add_argument('--iters',   type=int,   default=200)
    p.add_argument('--no_dye',  action='store_true',
                   help='disable the dye mix-norm scaling (4x factor)')
    p.add_argument('--device',  type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out',     type=str,
                   default='out/eulerian/halfdomain_sol.pt')
    args = p.parse_args()

    rho0 = half_domain_rho0(args.N_x, device=args.device)
    res = solve_eulerian(
        rho0, M_basis=args.M_basis, N_T=args.N_T, T=args.T,
        lam_mix=args.lam_mix, lam_kin=args.lam_kin,
        K_mix=args.K_mix, dye=not args.no_dye,
        iters=args.iters, device=args.device, verbose=True)

    print(f"final mix-norm^2   : {res['E_mix_curve'][-1]:.6e}")
    print(f"initial mix-norm^2 : {res['E_mix_curve'][0]:.6e}")
    print(f"E_kin              : {res['E_kin']:.6e}")
    print(f"mass drift         : "
          f"{abs(res['rho_traj'][-1].sum() - res['rho_traj'][0].sum()).item():.3e}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(res, args.out)
    print(f"Saved solution to {args.out}")


if __name__ == '__main__':
    main()
