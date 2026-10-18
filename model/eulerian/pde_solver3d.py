"""
pde_solver3d.py -- 3D port of pde_solver.py on Omega = (0,1)^3.

Transport rho under time-dependent control lam(t) in the div-free basis
from model.basis3d.DivFreeBasis3D (curl(psi e_k) with psi = sin.sin.sin),
same numerics as the 2D reference:
  * MAC staggered grid: cell-centred rho, face-centred v1/v2/v3
  * second-order MUSCL (minmod) upwind flux + Heun (RK2) in time
  * CFL-limited substepping
  * grid H^{-1} mix-norm via DCT-II on each of the three axes

Basis convention (MUST match basis3d.py, which omits the pi factors in
components and normalises with  c = 2 sqrt(2) / sqrt(i^2 + j^2)):

  axis k=1:  (0,  l sin(m pi x1) sin(n pi x2) cos(l pi x3),
                 -n sin(m pi x1) cos(n pi x2) sin(l pi x3))
  axis k=2:  (-l sin(m pi x1) sin(n pi x2) cos(l pi x3),  0,
                  m cos(m pi x1) sin(n pi x2) sin(l pi x3))
  axis k=3:  ( n sin(m pi x1) cos(n pi x2) sin(l pi x3),
                -m cos(m pi x1) sin(n pi x2) sin(l pi x3),  0)

Enumeration order for lam: (axis in {1,2,3}, m, n, l), matching
DivFreeBasis3D.__init__.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Div-free basis on the three MAC face grids
# ---------------------------------------------------------------------------

def _basis_on_faces_3d(M_basis: int, N_x: int,
                       device, dtype=torch.float64):
    """
    Pre-compute the div-free basis evaluated on the three MAC face grids.

    Face-coordinate layout (h = 1/N_x):
      u-face (x-face, component v1): x1 = i h,             x2 = (j+0.5) h,
                                      x3 = (k+0.5) h
      v-face (y-face, component v2): x1 = (i+0.5) h,       x2 = j h,
                                      x3 = (k+0.5) h
      w-face (z-face, component v3): x1 = (i+0.5) h,       x2 = (j+0.5) h,
                                      x3 = k h

    Returns (all float64)
      E1 : (L, N_x+1, N_x, N_x)
      E2 : (L, N_x,   N_x+1, N_x)
      E3 : (L, N_x,   N_x,   N_x+1)
    """
    pi = math.pi
    h  = 1.0 / N_x

    # Enumerate modes in the same order as DivFreeBasis3D.
    idx = [(k, m, n, l)
           for k in (1, 2, 3)
           for m in range(1, M_basis + 1)
           for n in range(1, M_basis + 1)
           for l in range(1, M_basis + 1)]
    L = len(idx)
    axis = torch.tensor([t[0] for t in idx], device=device, dtype=torch.int64)
    ms   = torch.tensor([t[1] for t in idx], device=device, dtype=torch.int64)
    ns   = torch.tensor([t[2] for t in idx], device=device, dtype=torch.int64)
    ls   = torch.tensor([t[3] for t in idx], device=device, dtype=torch.int64)

    pair_sq = torch.where(axis == 1, ns ** 2 + ls ** 2,
               torch.where(axis == 2, ms ** 2 + ls ** 2,
                                      ms ** 2 + ns ** 2)).to(dtype)
    norm = (2.0 * math.sqrt(2.0)) / torch.sqrt(pair_sq)              # (L,)

    # Face-grid 1-D coordinate arrays.
    x_int  = torch.arange(N_x + 1, device=device, dtype=dtype) * h    # (N_x+1,)
    x_mid  = (torch.arange(N_x, device=device, dtype=dtype) + 0.5) * h  # (N_x,)

    m_f = ms.to(dtype).view(L, 1, 1, 1)
    n_f = ns.to(dtype).view(L, 1, 1, 1)
    l_f = ls.to(dtype).view(L, 1, 1, 1)
    a1  = (axis == 1).to(dtype).view(L, 1, 1, 1)
    a2  = (axis == 2).to(dtype).view(L, 1, 1, 1)
    a3  = (axis == 3).to(dtype).view(L, 1, 1, 1)
    norm_f = norm.view(L, 1, 1, 1)

    # ---- u-face grid: x1 = i h (N_x+1),  x2 = (j+0.5) h,  x3 = (k+0.5) h -----
    #
    # At u-face we need the x-component (in e_x direction).  axis=1 contributes 0.
    x1u = x_int .view(1, N_x + 1, 1, 1)
    x2u = x_mid .view(1, 1, N_x, 1)
    x3u = x_mid .view(1, 1, 1, N_x)

    sm_u1 = torch.sin(m_f * pi * x1u)                # only sin for x1 (axis=2,3)
    sn_u2 = torch.sin(n_f * pi * x2u)
    cn_u2 = torch.cos(n_f * pi * x2u)
    sl_u3 = torch.sin(l_f * pi * x3u)
    cl_u3 = torch.cos(l_f * pi * x3u)
    # axis=2: -l sin(m x1) sin(n x2) cos(l x3)
    # axis=3:  n sin(m x1) cos(n x2) sin(l x3)
    P_ssc_u = sm_u1 * sn_u2 * cl_u3
    P_scs_u = sm_u1 * cn_u2 * sl_u3
    E1 = norm_f * ((n_f * P_scs_u) * a3 - (l_f * P_ssc_u) * a2)     # (L, Nx+1, Nx, Nx)
    # Analytic no-flow: sin(m pi 0) = sin(m pi 1) = 0 -> hard-zero at i=0, N_x.
    E1[:, 0,  :, :] = 0.0
    E1[:, -1, :, :] = 0.0

    # ---- v-face grid: x1 = (i+0.5) h,  x2 = j h (N_x+1),  x3 = (k+0.5) h ----
    x1v = x_mid .view(1, N_x, 1, 1)
    x2v = x_int .view(1, 1, N_x + 1, 1)
    x3v = x_mid .view(1, 1, 1, N_x)

    sm_v1 = torch.sin(m_f * pi * x1v)
    cm_v1 = torch.cos(m_f * pi * x1v)
    sn_v2 = torch.sin(n_f * pi * x2v)                # only sin (axis=1,3)
    sl_v3 = torch.sin(l_f * pi * x3v)
    cl_v3 = torch.cos(l_f * pi * x3v)
    # axis=1:  l sin(m x1) sin(n x2) cos(l x3)
    # axis=3: -m cos(m x1) sin(n x2) sin(l x3)
    P_ssc_v = sm_v1 * sn_v2 * cl_v3
    P_css_v = cm_v1 * sn_v2 * sl_v3
    E2 = norm_f * ((l_f * P_ssc_v) * a1 - (m_f * P_css_v) * a3)     # (L, Nx, Nx+1, Nx)
    E2[:, :, 0,  :] = 0.0
    E2[:, :, -1, :] = 0.0

    # ---- w-face grid: x1 = (i+0.5) h,  x2 = (j+0.5) h,  x3 = k h (N_x+1) ----
    x1w = x_mid .view(1, N_x, 1, 1)
    x2w = x_mid .view(1, 1, N_x, 1)
    x3w = x_int .view(1, 1, 1, N_x + 1)

    sm_w1 = torch.sin(m_f * pi * x1w)
    cm_w1 = torch.cos(m_f * pi * x1w)
    sn_w2 = torch.sin(n_f * pi * x2w)
    cn_w2 = torch.cos(n_f * pi * x2w)
    sl_w3 = torch.sin(l_f * pi * x3w)                # only sin (axis=1,2)
    # axis=1: -n sin(m x1) cos(n x2) sin(l x3)
    # axis=2:  m cos(m x1) sin(n x2) sin(l x3)
    P_scs_w = sm_w1 * cn_w2 * sl_w3
    P_css_w = cm_w1 * sn_w2 * sl_w3
    E3 = norm_f * ((m_f * P_css_w) * a2 - (n_f * P_scs_w) * a1)     # (L, Nx, Nx, Nx+1)
    E3[:, :, :, 0]  = 0.0
    E3[:, :, :, -1] = 0.0

    return E1, E2, E3


def _assemble_velocity_3d(lam_k: torch.Tensor,
                          E1: torch.Tensor,
                          E2: torch.Tensor,
                          E3: torch.Tensor):
    """lam_k: (L,) -> (v1, v2, v3) on the three face grids."""
    v1 = torch.einsum('l,lijk->ijk', lam_k, E1)
    v2 = torch.einsum('l,lijk->ijk', lam_k, E2)
    v3 = torch.einsum('l,lijk->ijk', lam_k, E3)
    return v1, v2, v3


# ---------------------------------------------------------------------------
# MUSCL (minmod) upwind transport in 3D
# ---------------------------------------------------------------------------

def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    same = (a * b) > 0.0
    mag  = torch.minimum(a.abs(), b.abs())
    return torch.where(same, torch.sign(a) * mag, torch.zeros_like(a))


def _advect_rhs_3d(rho: torch.Tensor,
                   v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor,
                   h: float) -> torch.Tensor:
    """
    -(dF1/dx + dF2/dy + dF3/dz)/h   shape (Nx, Nx, Nx).

    Layout:
      rho : (Nx, Nx, Nx)     cell centres
      v1  : (Nx+1, Nx, Nx)   u-face (x-face)
      v2  : (Nx, Nx+1, Nx)   v-face (y-face)
      v3  : (Nx, Nx, Nx+1)   w-face (z-face)
    """
    Nx = rho.shape[0]

    # ---- x-direction -------------------------------------------------------
    rho_pad = F.pad(rho, (0, 0, 0, 0, 1, 1))                # pad axis 0
    diff_fwd = rho_pad[2:,   :, :] - rho_pad[1:-1, :, :]    # rho[i+1]-rho[i]
    diff_bwd = rho_pad[1:-1, :, :] - rho_pad[:-2,  :, :]    # rho[i]-rho[i-1]
    slope_x  = _minmod(diff_fwd, diff_bwd)                   # (Nx, Nx, Nx)
    rho_L = rho - 0.5 * slope_x
    rho_R = rho + 0.5 * slope_x

    u_int = v1[1:-1, :, :]                                   # (Nx-1, Nx, Nx)
    face_left  = rho_R[:-1, :, :]
    face_right = rho_L[1:,  :, :]
    rho_face_x = torch.where(u_int >= 0.0, face_left, face_right)
    F1_int = u_int * rho_face_x

    F1 = rho.new_zeros(Nx + 1, Nx, Nx)
    F1[1:-1, :, :] = F1_int

    # ---- y-direction -------------------------------------------------------
    rho_pad = F.pad(rho, (0, 0, 1, 1, 0, 0))                # pad axis 1
    diff_fwd = rho_pad[:, 2:,   :] - rho_pad[:, 1:-1, :]
    diff_bwd = rho_pad[:, 1:-1, :] - rho_pad[:, :-2,  :]
    slope_y  = _minmod(diff_fwd, diff_bwd)
    rho_Ly = rho - 0.5 * slope_y
    rho_Ry = rho + 0.5 * slope_y

    w_int = v2[:, 1:-1, :]                                   # (Nx, Nx-1, Nx)
    face_below = rho_Ry[:, :-1, :]
    face_above = rho_Ly[:, 1:,  :]
    rho_face_y = torch.where(w_int >= 0.0, face_below, face_above)
    F2_int = w_int * rho_face_y

    F2 = rho.new_zeros(Nx, Nx + 1, Nx)
    F2[:, 1:-1, :] = F2_int

    # ---- z-direction -------------------------------------------------------
    rho_pad = F.pad(rho, (1, 1, 0, 0, 0, 0))                # pad axis 2
    diff_fwd = rho_pad[:, :, 2:]   - rho_pad[:, :, 1:-1]
    diff_bwd = rho_pad[:, :, 1:-1] - rho_pad[:, :, :-2]
    slope_z  = _minmod(diff_fwd, diff_bwd)
    rho_Lz = rho - 0.5 * slope_z
    rho_Rz = rho + 0.5 * slope_z

    q_int = v3[:, :, 1:-1]                                   # (Nx, Nx, Nx-1)
    face_back  = rho_Rz[:, :, :-1]
    face_front = rho_Lz[:, :, 1:]
    rho_face_z = torch.where(q_int >= 0.0, face_back, face_front)
    F3_int = q_int * rho_face_z

    F3 = rho.new_zeros(Nx, Nx, Nx + 1)
    F3[:, :, 1:-1] = F3_int

    div = (F1[1:, :, :] - F1[:-1, :, :]
         + F2[:, 1:, :] - F2[:, :-1, :]
         + F3[:, :, 1:] - F3[:, :, :-1]) / h
    return -div


def advect_step_3d(rho, v1, v2, v3, dt, h):
    """One Heun (RK2) step."""
    r1 = _advect_rhs_3d(rho, v1, v2, v3, h)
    r2 = _advect_rhs_3d(rho + dt * r1, v1, v2, v3, h)
    return rho + 0.5 * dt * (r1 + r2)


def _max_abs_velocity_3d(v1, v2, v3):
    return torch.maximum(torch.maximum(v1.abs().max(), v2.abs().max()),
                         v3.abs().max())


def advect_interval_3d(rho, v1, v2, v3, dt, h, cfl: float = 0.5):
    """Advance rho by dt under (v1,v2,v3) with CFL-limited substepping."""
    vmax = _max_abs_velocity_3d(v1, v2, v3)
    if vmax.item() * dt / h <= cfl or vmax.item() == 0.0:
        return advect_step_3d(rho, v1, v2, v3, dt, h)

    n_sub = int(math.ceil(vmax.item() * dt / (cfl * h)))
    sub_dt = dt / n_sub
    out = rho
    for _ in range(n_sub):
        out = advect_step_3d(out, v1, v2, v3, sub_dt, h)
    return out


# ---------------------------------------------------------------------------
# Grid H^{-1} mix-norm via DCT-II (Neumann cosine basis)
# ---------------------------------------------------------------------------

def _dct1(u: torch.Tensor) -> torch.Tensor:
    """Type-II DCT along the last axis (unnormalised, scipy-style)."""
    N = u.shape[-1]
    y = torch.cat([u, u.flip(-1)], dim=-1)                       # length 2N
    Y = torch.fft.rfft(y, n=2 * N, dim=-1)                        # (..., N+1)
    k = torch.arange(N + 1, device=u.device, dtype=u.dtype)
    phase = torch.exp(-1j * math.pi * k / (2 * N))
    return (Y * phase).real[..., :N]


def _dct3_3d(x: torch.Tensor) -> torch.Tensor:
    """Type-II DCT along the three axes."""
    y = _dct1(x)
    y = _dct1(y.transpose(-1, -2)).transpose(-1, -2)
    y = _dct1(y.transpose(-1, -3)).transpose(-1, -3)
    return y


def grid_mix_norm_sq_3d(rho: torch.Tensor,
                        K_max: int | None = None,
                        dye: bool = False) -> torch.Tensor:
    """
    Discrete ||rho - mean(rho)||^2_{H^{-1}((0,1)^3)} on an N_x^3 cell-centred
    grid, matching basis3d.SpectralMixingCost3D (Neumann cosine basis).
    """
    Nx = rho.shape[-1]
    h  = 1.0 / Nx
    coeff = _dct3_3d(rho.to(torch.float64))                       # (Nx,Nx,Nx)

    device, dtype = rho.device, torch.float64
    k = torch.arange(Nx, device=device, dtype=dtype)
    a = torch.where(k == 0,
                    torch.ones_like(k),
                    math.sqrt(2.0) * torch.ones_like(k))

    ak = a[:, None, None] * a[None, :, None] * a[None, None, :]    # (Nx,Nx,Nx)
    # 3D DCT-II carries a factor 8 = 2^3 relative to the quadrature sum.
    rho_hat = (h ** 3 / 8.0) * ak * coeff

    pi = math.pi
    eigs = (pi ** 2) * (k[:, None, None] ** 2
                       + k[None, :, None] ** 2
                       + k[None, None, :] ** 2)
    eigs[0, 0, 0] = 1.0

    weight = 1.0 / eigs
    weight[0, 0, 0] = 0.0
    if K_max is not None:
        mask = (k[:, None, None] <= K_max) \
             & (k[None, :, None] <= K_max) \
             & (k[None, None, :] <= K_max)
        weight = weight * mask.to(weight.dtype)

    out = (rho_hat * rho_hat * weight).sum().to(rho.dtype)
    if dye:
        out = 4.0 * out
    return out


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def simulate_3d(lam: torch.Tensor,
                rho0: torch.Tensor,
                E1: torch.Tensor,
                E2: torch.Tensor,
                E3: torch.Tensor,
                T: float,
                cfl: float = 0.5,
                record_rho: bool = False):
    """Integrate rho under the time-dependent basis control lam (N_T, L)."""
    N_T = lam.shape[0]
    Nx  = rho0.shape[0]
    h   = 1.0 / Nx
    dt  = T / N_T

    rho = rho0
    if record_rho:
        traj = [rho.detach().clone()]
    for k in range(N_T):
        v1, v2, v3 = _assemble_velocity_3d(lam[k], E1, E2, E3)
        rho = advect_interval_3d(rho, v1, v2, v3, dt, h, cfl=cfl)
        if record_rho:
            traj.append(rho.detach().clone())
    if record_rho:
        return rho, torch.stack(traj, dim=0)
    return rho, None


# ---------------------------------------------------------------------------
# Family-aware initial density
# ---------------------------------------------------------------------------

def family_rho0_grid_3d(family, p, N_x: int, device,
                        dtype=torch.float64) -> torch.Tensor:
    """rho0[i,j,k] = 1 iff family.predicate holds at cell centre."""
    g = (torch.arange(N_x, device=device, dtype=dtype) + 0.5) / N_x
    x1, x2, x3 = torch.meshgrid(g, g, g, indexing='ij')
    pts = torch.stack([x1.reshape(-1), x2.reshape(-1), x3.reshape(-1)], dim=-1)
    mask = family.predicate(pts, p.to(device=device, dtype=dtype))
    return mask.to(dtype).reshape(N_x, N_x, N_x)


def half_domain_rho0_3d(N_x: int, device, dtype=torch.float64) -> torch.Tensor:
    """rho_0(x) = 1 if x1 < 0.5 else 0 on the cell-centred cube grid."""
    h  = 1.0 / N_x
    x1 = (torch.arange(N_x, device=device, dtype=dtype) + 0.5) * h
    out = torch.zeros(N_x, N_x, N_x, device=device, dtype=dtype)
    out[x1 < 0.5, :, :] = 1.0
    return out
