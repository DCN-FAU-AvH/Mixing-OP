"""
compare_pde_particle.py -- PDE-level comparison of the neural operator
against the Eulerian optimum.

BOTH velocity fields are evaluated through the SAME Eulerian transport
solver (same MAC grid, same MUSCL + RK2, same dye mix-norm) so the
comparison is apples-to-apples at the PDE level.  The only difference
between the two columns is the time-dependent control lambda(t):
  - lambda^NN(t): read off the trained MixingOperatorNet at N_T midpoints
  - lambda^Eul(t): optimised by solve_eulerian from pde_solver.py

Half-domain initial density  rho_0(x) = 1  iff x1 < 0.5.

Outputs (matching the visualization style of example.test_multi)
---------------------------------------------------------------------
  out/eulerian/compare_halfdomain_multi.png
      2 rows (Eulerian / Network), each row:
        - 6 snapshot panels (red/blue density on the grid) at t=0..T
        - 1 log-scale mix-norm decay panel on the right

  out/eulerian/compare_halfdomain_lambda.png
      2 subplots (Eulerian / Network), lambda_{mn}(t) curves.

  out/eulerian/compare_halfdomain.pt   raw tensors for downstream use.
"""

from __future__ import annotations

# Allow both module execution and direct execution of this example.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import math
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress

from model.basis import DivFreeBasis
from model   import MixingOperatorNet
from train   import _lam_at
from model.dataset import build_family

from model.eulerian.pde_solver import (
    solve_eulerian, half_domain_rho0, grid_mix_norm_sq,
    _basis_on_faces, _assemble_velocity, advect_interval,
)


# ---------------------------------------------------------------------------
# Family-aware IC builders (replace the old half-domain helpers)
# ---------------------------------------------------------------------------

def family_rho0_grid(family, p, N_x: int, device: str,
                     dtype=torch.float64) -> torch.Tensor:
    """rho0[i,j] = 1 iff predicate holds at cell centre ((i+0.5)h, (j+0.5)h)."""
    g = (torch.arange(N_x, device=device, dtype=dtype) + 0.5) / N_x
    x1, x2 = torch.meshgrid(g, g, indexing='ij')
    pts = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)
    mask = family.predicate(pts, p.to(device=device, dtype=dtype))
    return mask.to(dtype).reshape(N_x, N_x)


def family_lattice(family, p, N_grid: int,
                   device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform lattice split into red (predicate True) and blue (else)."""
    g = torch.linspace(0.5 / N_grid, 1.0 - 0.5 / N_grid, N_grid, device=device)
    x1, x2 = torch.meshgrid(g, g, indexing='ij')
    pts = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)
    mask = family.predicate(pts, p.to(device=device, dtype=pts.dtype))
    X1 = pts[mask].unsqueeze(0)
    X2 = pts[~mask].unsqueeze(0)
    return X1, X2


# ---------------------------------------------------------------------------
# Checkpoint loading (pattern from test_multi.load_model)
# ---------------------------------------------------------------------------

def load_network(ckpt_path: str, device: str):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train a model with train.py or pass an existing file with "
            "--checkpoint.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get('args', {})

    basis = DivFreeBasis(a.get('M_basis', 2)).to(device)
    net = MixingOperatorNet(
        d              = 2,
        L              = basis.L,
        h              = a.get('h', 256),
        n_heads        = a.get('n_heads', 4),
        n_layers_self  = a.get('n_layers_self', 1),
        n_layers_joint = a.get('n_layers_joint', 1),
        t_emb_dim      = a.get('t_emb_dim', 64),
    ).to(device)
    net.load_state_dict(ckpt['model'])
    net.eval()
    return net, basis, a


# ---------------------------------------------------------------------------
# Particle initial condition: uniform lattice, split red (x1<0.5) / blue
# (matching model.util.make_lattice which test_multi also uses).
# ---------------------------------------------------------------------------

def halfdomain_lattice(N_grid: int, device: str):
    g = torch.linspace(0.5 / N_grid, 1.0 - 0.5 / N_grid, N_grid, device=device)
    x1, x2 = torch.meshgrid(g, g, indexing='ij')
    pts = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)
    mask = pts[:, 0] < 0.5
    X1 = pts[mask].unsqueeze(0)      # red
    X2 = pts[~mask].unsqueeze(0)     # blue
    return X1, X2


# ---------------------------------------------------------------------------
# Query the network for lambda(t) at N_T control midpoints
# ---------------------------------------------------------------------------

@torch.no_grad()
def network_lambda_schedule(net, X1_0, X2_0, T, N_T):
    """
    Evaluate lambda^NN(t) at the N_T ODE sub-interval midpoints so the
    resulting schedule plugs directly into the Eulerian transport solver.

    Returns lam_nn : (N_T, L)
    """
    z = net.encode_context(X1_0, X2_0)
    dt = T / N_T
    lams = []
    for s in range(N_T):
        lam_s = _lam_at(net, z, s * dt + 0.5 * dt, T)     # (1, L)
        lams.append(lam_s[0].cpu())
    return torch.stack(lams, dim=0)


# ---------------------------------------------------------------------------
# Eulerian simulation with the same fixed lambda returned by solve_eulerian
# ---------------------------------------------------------------------------

def simulate_eulerian_full(lam: torch.Tensor, rho0: torch.Tensor,
                           E1, E2, T: float, cfl: float = 0.5):
    """
    Transport rho0 under the time-dependent control lam through the same
    Eulerian solver (MUSCL + RK2) used by the PDE optimiser.

    Returns
    -------
    rho_traj : list of length N_T+1 of (Nx, Nx) tensors
    times    : (N_T+1,) np array
    """
    N_T = lam.shape[0]
    Nx  = rho0.shape[0]
    h   = 1.0 / Nx
    dt  = T / N_T

    rho = rho0.clone()
    traj  = [rho.clone()]
    times = [0.0]
    for k in range(N_T):
        v1, v2 = _assemble_velocity(lam[k], E1, E2)
        rho = advect_interval(rho, v1, v2, dt, h, cfl=cfl)
        traj.append(rho.clone())
        times.append((k + 1) * dt)
    return traj, np.asarray(times)


# ---------------------------------------------------------------------------
# Helpers to add an exponential-fit annotation on the mix-norm decay panel
# (mirrors test_multi.py linregress block)
# ---------------------------------------------------------------------------

def _exp_fit(t_arr: np.ndarray, mix_arr: np.ndarray):
    """Fit  mix(t) ~ k * exp(b * t)  over the last 75% of the trajectory."""
    idx = max(1, int(0.25 * len(t_arr)))
    valid = mix_arr[idx:] > 0
    if valid.sum() < 2:
        return None
    t_fit  = t_arr[idx:][valid]
    log_mx = np.log(mix_arr[idx:][valid])
    slope, intercept, *_ = linregress(t_fit, log_mx)
    return float(np.exp(intercept)), float(slope)


# ---------------------------------------------------------------------------
# Figure 1: snapshot grid + log-scale decay, 2 rows (Eul, NN)
# ---------------------------------------------------------------------------

def fig_multi(eul_snaps, eul_times, nn_snaps, nn_times,
              eul_mix_curve, nn_mix_curve,
              eul_mix_times, nn_mix_times,
              n_panels: int, out_path: str):
    """Imitates the layout of test_multi.py's Figure 1."""
    n_rows = 2
    cell   = 1.8
    g_col  = 0.08
    g_row  = 0.08
    g_mid  = 1.2
    pad_l, pad_r, pad_t, pad_b = 0.05, 0.20, 0.35, 0.35

    blk_w  = n_panels * cell + (n_panels - 1) * g_col
    fig_w  = pad_l + blk_w + g_mid + cell + pad_r
    fig_h  = pad_b + n_rows * cell + (n_rows - 1) * g_row + pad_t

    fig = plt.figure(figsize=(fig_w, fig_h))

    L_left   = pad_l / fig_w
    R_left   = (pad_l + blk_w) / fig_w
    L_right  = (pad_l + blk_w + g_mid) / fig_w
    R_right  = (fig_w - pad_r) / fig_w
    gs_B     = pad_b / fig_h
    gs_T     = (fig_h - pad_t) / fig_h
    hs_frac  = g_row / cell
    ws_frac  = g_col / cell

    gs_left = gridspec.GridSpec(
        n_rows, n_panels,
        left=L_left, right=R_left, bottom=gs_B, top=gs_T,
        hspace=hs_frac, wspace=ws_frac)

    gs_right = gridspec.GridSpec(
        n_rows, 1,
        left=L_right, right=R_right, bottom=gs_B, top=gs_T,
        hspace=hs_frac)

    # Row labels + colour scheme.  Both rows are grid densities produced by
    # the SAME Eulerian transport solver; the only difference is the lambda
    # schedule (Eulerian-optimised vs network-predicted).
    rows = [
        ('Eulerian',   eul_snaps, eul_times, '#1a66cc',
         eul_mix_times, eul_mix_curve),
        ('Network',    nn_snaps,  nn_times,  '#cc3311',
         nn_mix_times,  nn_mix_curve),
    ]

    def _sel_idx(n_snaps, n_panels):
        if n_snaps <= n_panels:
            return list(range(n_snaps))
        sel = [int(round(i * (n_snaps - 1) / (n_panels - 1)))
               for i in range(n_panels)]
        sel[0] = 0; sel[-1] = n_snaps - 1
        return sel

    red_rgb  = np.array([0.84, 0.15, 0.16])
    blue_rgb = np.array([0.12, 0.47, 0.71])

    for row, (name, snaps, times, color, mix_t, mix_v) in enumerate(rows):
        sel = _sel_idx(len(snaps), n_panels)

        for col_idx, si in enumerate(sel):
            ax = fig.add_subplot(gs_left[row, col_idx])
            rho = snaps[si].cpu().numpy() if torch.is_tensor(snaps[si]) \
                  else snaps[si]
            a = np.clip(rho[..., None], 0.0, 1.0)
            rgb = a * red_rgb + (1 - a) * blue_rgb
            ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower',
                      extent=[0, 1, 0, 1], aspect='equal')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f't = {times[si]:.1f}', fontsize=14, pad=4)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)
            if col_idx == 0:
                ax.set_ylabel(name, fontsize=12, color=color,
                              fontweight='bold')

        # mix-norm decay (log-y) with exponential fit
        ax_norm = fig.add_subplot(gs_right[row, 0])
        ax_norm.set_box_aspect(1)
        t_arr   = np.asarray(mix_t)
        mix_arr = np.sqrt(np.maximum(np.asarray(mix_v), 0.0))
        ax_norm.plot(t_arr, mix_arr, color=color, linewidth=2,
                     label=r'$\Vert \rho - \bar\rho \Vert_{\dot{H}^{-1}}$')
        fit = _exp_fit(t_arr, mix_arr)
        if fit is not None:
            k_fit, b_fit = fit
            ax_norm.plot(t_arr, k_fit * np.exp(b_fit * t_arr), 'k--',
                         linewidth=1.5,
                         label=fr'${k_fit:.2f}\times e^{{{b_fit:.2f}\,t}}$')
        ax_norm.set_yscale('log')
        if row == n_rows - 1:
            ax_norm.set_xlabel('t', fontsize=11)
        else:
            ax_norm.tick_params(labelbottom=False)
        ax_norm.set_ylabel(r'$\Vert \rho - \bar\rho \Vert_{\dot{H}^{-1}}$',
                           fontsize=10)
        ax_norm.tick_params(which='both', labelsize=8)
        ax_norm.margins(x=0)
        ax_norm.legend(fontsize=7, loc='best')
        if row == 0:
            ax_norm.set_title('mix-norm', fontsize=14, pad=4)

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: lambda curves, 2 subplots (Eul / NN)
# ---------------------------------------------------------------------------

def fig_lambda(lam_eul: torch.Tensor, t_eul: np.ndarray,
               lam_nn:  torch.Tensor, t_nn:  np.ndarray,
               ms_np, ns_np, M_basis: int, T: float, out_path: str,
               title_tag: str = 'half-domain'):
    L = lam_eul.shape[1]
    cmap  = plt.cm.tab20 if L > 10 else plt.cm.tab10

    fig2, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    panels = [
        ('Eulerian', lam_eul, t_eul, '#1a66cc'),
        ('Network',  lam_nn,  t_nn,  '#cc3311'),
    ]
    for ax, (tag, lam, t, col) in zip(axes, panels):
        for l_idx in range(L):
            c = cmap(l_idx % 20)
            lbl = f'$\\lambda_{{{ms_np[l_idx]},{ns_np[l_idx]}}}$' \
                  if L <= 12 else None
            ax.plot(t, lam[:, l_idx].numpy(), color=c, linewidth=1.4,
                    alpha=0.9, label=lbl)
        ax.set_title(tag, fontsize=11, fontweight='bold', color=col)
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel(r'$\lambda_\ell(t)$', fontsize=10)
        ax.grid(True, alpha=0.3)
        if L <= 12:
            ax.legend(fontsize=7, ncol=2, loc='best')

    fig2.suptitle(
        f'$\\lambda(t)$  {title_tag}  (M={M_basis}, L={L}, T={T})',
        fontsize=12, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="PDE vs network comparison")
    p.add_argument('--checkpoint', type=str,
                   default='out/rotation_line_d1/ckpt_final.pt')
    p.add_argument('--out_dir',    type=str, default='out/eulerian')
    p.add_argument('--N_x',        type=int, default=128)
    p.add_argument('--N_T',        type=int, default=None,
                   help='SHARED Eulerian/lambda time-step count (default = ckpt N_T). '
                        'Both NN and Eulerian lambdas are sampled at the same '
                        'N_T midpoints and fed through the same transport solver.')
    p.add_argument('--N_grid_ctx', type=int, default=50,
                   help='lattice size for the NN context (red+blue particles) '
                        'on which encode_context is called')
    p.add_argument('--n_panels',   type=int, default=6)
    p.add_argument('--T',          type=float, default=None,
                   help='time horizon; default = ckpt T')
    p.add_argument('--M_basis_eul', type=int, default=None,
                   help='basis order for the Eulerian solver; '
                        'default = ckpt M_basis')
    p.add_argument('--K_mix',       type=int, default=None,
                   help='spectral truncation; default = ckpt K_mix')
    p.add_argument('--lam_mix',    type=float, default=None,
                   help='mix-norm weight; default = ckpt lam_mix')
    p.add_argument('--lam_kin',    type=float, default=None,
                   help='kinetic weight; default = ckpt lam_kin')
    p.add_argument('--iters',      type=int,   default=200)
    p.add_argument('--device',     type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--family',     type=str, default=None,
                   help="Partition family name. Default = ckpt['family'].")
    p.add_argument('--param_mode', type=str, default='test',
                   choices=['train', 'test'],
                   help='Sample param from inside (test) or outside (train) '
                        'the generalisation hole.')
    p.add_argument('--param',      type=str, default=None,
                   help='Comma-separated parameter values overriding sampling.')
    p.add_argument('--tag',        type=str, default=None,
                   help='Filename stem; default = family name.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # ------------------------------------------------------------------ net
    print(f"Loading checkpoint: {args.checkpoint}")
    net, basis, ckpt_args = load_network(args.checkpoint, device)
    T        = args.T       if args.T       is not None else ckpt_args.get('T', 5.0)
    M_basis  = args.M_basis_eul if args.M_basis_eul is not None else ckpt_args.get('M_basis', 2)
    N_T      = args.N_T     if args.N_T     is not None else ckpt_args.get('N_T', 20)
    K_mix    = args.K_mix   if args.K_mix   is not None else ckpt_args.get('K_mix', 12)
    lam_mix  = args.lam_mix if args.lam_mix is not None else ckpt_args.get('lam_mix', 1.0)
    lam_kin  = args.lam_kin if args.lam_kin is not None else ckpt_args.get('lam_kin', 1e-3)
    L        = basis.L
    print(f"  Matching training hparams: M_basis={M_basis}  L={L}  T={T}  "
          f"N_T={N_T}  K_mix={K_mix}  "
          f"lam_mix={lam_mix}  lam_kin={lam_kin}  (dye mix-norm)")

    # ------------------------------------------------------------------ family + IC
    family_name = args.family if args.family is not None \
                  else ckpt_args.get('family', 'rotation_line')
    level_set_K = ckpt_args.get('level_set_K', 3)
    family = build_family(family_name, level_set_K=level_set_K)
    tag = args.tag if args.tag is not None else family_name

    torch.manual_seed(args.seed)
    if args.param is not None:
        p_vec = torch.tensor([float(x) for x in args.param.split(',')],
                             dtype=torch.float32, device=device)
        if p_vec.numel() != family.dim:
            raise ValueError(
                f"--param has {p_vec.numel()} values but family "
                f"'{family_name}' expects {family.dim}.")
    else:
        p_vec = family.sample_param(mode=args.param_mode, device=device)
    print(f"Family: {family_name}  param = {family.describe(p_vec)}")

    Nx   = args.N_x
    dtype = torch.float64
    rho0 = family_rho0_grid(family, p_vec, Nx, device=device, dtype=dtype)
    E1, E2 = _basis_on_faces(M_basis, Nx, device=device, dtype=dtype)

    # ------------------------------------------------------------------ NN lambda
    print(f"Reading lambda^NN from the trained operator (N_grid_ctx="
          f"{args.N_grid_ctx}^2 context particles, N_T={N_T})...")
    X1_0, X2_0 = family_lattice(family, p_vec, args.N_grid_ctx, device)
    lam_nn = network_lambda_schedule(net, X1_0, X2_0, T, N_T)

    # ------------------------------------------------------------------ Eulerian optimum
    print(f"Optimising lambda^Eul: N_x={Nx}, N_T={N_T}, iters={args.iters}...")
    eul = solve_eulerian(
        rho0, M_basis=M_basis, N_T=N_T, T=T,
        lam_mix=lam_mix, lam_kin=lam_kin,
        K_mix=K_mix, dye=True,
        iters=args.iters, device=device, verbose=True)
    lam_eul = eul['lam']

    # ------------------------------------------------------------------ PDE-level rollouts
    # Push rho0 through the SAME transport solver for both lambdas.
    print("Pushing both lambda schedules through the Eulerian transport solver...")
    nn_traj, nn_mix_times = simulate_eulerian_full(
        lam_nn.to(device=device, dtype=dtype),
        rho0.to(dtype=dtype), E1, E2, T)
    eul_traj, eul_mix_times = simulate_eulerian_full(
        lam_eul.to(device=device, dtype=dtype),
        rho0.to(dtype=dtype), E1, E2, T)

    # ------------------------------------------------------------------ mix-norm curves
    # Identical cost (dye + K_mix truncation) applied to both trajectories.
    with torch.no_grad():
        eul_mix_curve = np.asarray(
            [float(grid_mix_norm_sq(r, K_max=K_mix, dye=True))
             for r in eul_traj])
        nn_mix_curve  = np.asarray(
            [float(grid_mix_norm_sq(r, K_max=K_mix, dye=True))
             for r in nn_traj])

    # Snapshot frames for the figure -- pick n_panels evenly spaced indices.
    nn_snaps, nn_times   = nn_traj,  nn_mix_times
    eul_snaps, eul_times = eul_traj, eul_mix_times

    # ------------------------------------------------------------------ report
    dt = T / N_T
    kin_nn  = float((dt * (lam_nn ** 2).sum()).item())
    kin_eul = float((dt * (lam_eul ** 2).sum()).item())
    mass0 = float(eul_traj[0].sum().item()) * (1.0 / Nx) ** 2
    mass_eulT = float(eul_traj[-1].sum().item()) * (1.0 / Nx) ** 2
    mass_nnT  = float(nn_traj[-1].sum().item())  * (1.0 / Nx) ** 2
    print("\n==== PDE-level scalar report ====")
    print(f"  E_mix^0    : {eul_mix_curve[0]:.4e}  (shared)")
    print(f"  E_mix^T    : Eul {eul_mix_curve[-1]:.4e}   NN {nn_mix_curve[-1]:.4e}")
    print(f"  E_kin (dt * sum lam^2)")
    print(f"             : Eul {kin_eul:.4e}             NN {kin_nn:.4e}")
    print(f"  mass       : Eul {mass0:.6f} -> {mass_eulT:.6f}  "
          f"(drift {mass_eulT - mass0:.2e})")
    print(f"               NN  {mass0:.6f} -> {mass_nnT:.6f}  "
          f"(drift {mass_nnT  - mass0:.2e})")

    # ------------------------------------------------------------------ figures
    fig_multi(
        eul_snaps, eul_times, nn_snaps, nn_times,
        eul_mix_curve, nn_mix_curve,
        eul_mix_times, nn_mix_times,
        n_panels=args.n_panels,
        out_path=os.path.join(args.out_dir, f'compare_{tag}_multi.png'),
    )

    ms_np = basis.ms.cpu().numpy().astype(int)
    ns_np = basis.ns.cpu().numpy().astype(int)
    t_mid = np.linspace(0.5 * (T / N_T),
                        T - 0.5 * (T / N_T), N_T)
    fig_lambda(
        lam_eul, t_mid, lam_nn, t_mid,
        ms_np, ns_np, M_basis, T,
        out_path=os.path.join(args.out_dir, f'compare_{tag}_lambda.png'),
        title_tag=family.describe(p_vec),
    )

    # Raw dump
    dump_path = os.path.join(args.out_dir, f'compare_{tag}.pt')
    torch.save({
        'lam_eul':          lam_eul,
        'lam_nn':           lam_nn,
        'rho_eul_traj':     torch.stack([r.cpu() for r in eul_traj]),
        'rho_nn_traj':      torch.stack([r.cpu() for r in nn_traj]),
        'eul_mix_curve':    torch.from_numpy(eul_mix_curve),
        'nn_mix_curve':     torch.from_numpy(nn_mix_curve),
        'times':            torch.from_numpy(eul_mix_times),
        'rho0':             rho0.cpu(),
        'E_kin_eul':        kin_eul,
        'E_kin_nn':         kin_nn,
        'meta':             eul['meta'],
        'ckpt_args':        ckpt_args,
        'family':           family_name,
        'family_param':     p_vec.detach().cpu(),
    }, dump_path)
    print(f"Saved: {dump_path}")


if __name__ == '__main__':
    main()
