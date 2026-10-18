"""
compare_particle_pde.py

Two-row verification figure per family:

  row 1  particle rollout of the network velocity field
         (scatter of red/blue particles at 6 time snapshots + mix-norm from
          the empirical dye cost used at training time)

  row 2  Eulerian PDE rollout of the SAME network velocity field on a high
         resolution MAC grid N_x x N_x, with mix-norm computed by a DCT
         truncation that is the grid counterpart of the eigenfunction sum
         used for the particle cost.

The two rows share the same initial density, the same time horizon T, and
the same time-dependent basis coefficient vector lambda^NN(t) read off the
trained operator.  They therefore check that the Lagrangian particle
rollout and the Eulerian PDE rollout see the same mixing action.

Usage
-----
    python -m example.compare_particle_pde \
        --checkpoint out/rotation_line_d1/ckpt_final.pt \
        --out_dir    out/eulerian_hires \
        --N_x        256

One figure is produced per call.  A driver loop over the four 2D families
lives at the bottom of the file (run via --all).
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

from model.basis import DivFreeBasis, DyeMixCost
from model   import MixingOperatorNet
from train   import _lam_at
from model.dataset import build_family

from model.eulerian.pde_solver import (
    grid_mix_norm_sq, _basis_on_faces, _assemble_velocity, advect_interval,
)

from example.compare_pde_particle import (
    family_rho0_grid, family_lattice, load_network,
    network_lambda_schedule, simulate_eulerian_full, _exp_fit,
)


# ---------------------------------------------------------------------------
# Particle rollout with snapshots (mirrors test.integrate_with_snapshots)
# ---------------------------------------------------------------------------

@torch.no_grad()
def particle_rollout(net, basis, z, X1, X2, T, N_T, N_snap):
    """RK4 rollout of the network velocity on two particle groups.

    Returns snapshots = [(X1_s, X2_s), ...] of length N_snap + 1,
            snap_times.
    """
    device = X1.device
    dt = T / N_T
    X1_cur = X1.clone()
    X2_cur = X2.clone()

    snap_stride = max(1, N_T // N_snap)
    snaps = [(X1_cur.clone(), X2_cur.clone())]
    snap_times = [0.0]

    for s in range(N_T):
        t0 = s * dt
        lam1 = _lam_at(net, z, t0,             T)
        lam2 = _lam_at(net, z, t0 + 0.5 * dt,  T)
        lam4 = _lam_at(net, z, t0 + dt,        T)

        def vel(X, lam):
            return basis.velocity(X, lam)

        k1a = vel(X1_cur, lam1);                 k1b = vel(X2_cur, lam1)
        k2a = vel(X1_cur + 0.5*dt*k1a, lam2);    k2b = vel(X2_cur + 0.5*dt*k1b, lam2)
        k3a = vel(X1_cur + 0.5*dt*k2a, lam2);    k3b = vel(X2_cur + 0.5*dt*k2b, lam2)
        k4a = vel(X1_cur +     dt*k3a, lam4);    k4b = vel(X2_cur +     dt*k3b, lam4)

        X1_cur = X1_cur + (dt / 6.0) * (k1a + 2*k2a + 2*k3a + k4a)
        X2_cur = X2_cur + (dt / 6.0) * (k1b + 2*k2b + 2*k3b + k4b)
        X1_cur = X1_cur.clamp(0.0, 1.0)
        X2_cur = X2_cur.clamp(0.0, 1.0)

        if (s + 1) % snap_stride == 0 or s == N_T - 1:
            snaps.append((X1_cur.clone(), X2_cur.clone()))
            snap_times.append((s + 1) * dt)

    return snaps, snap_times


def _sel_idx(n_snaps, n_panels):
    if n_snaps <= n_panels:
        return list(range(n_snaps))
    sel = [int(round(i * (n_snaps - 1) / (n_panels - 1))) for i in range(n_panels)]
    sel[0] = 0; sel[-1] = n_snaps - 1
    return sel


# ---------------------------------------------------------------------------
# Figure: two rows (particle, PDE) + mix-norm column
# ---------------------------------------------------------------------------

def make_figure(particle_snaps, particle_times, particle_mix,
                pde_snaps, pde_times, pde_mix,
                out_path, n_panels=6):
    n_rows = 2
    cell   = 1.8
    g_col  = 0.08
    g_row  = 0.08
    g_mid  = 1.2
    pad_l, pad_r, pad_t, pad_b = 0.08, 0.25, 0.35, 0.45

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

    red_rgb  = np.array([0.84, 0.15, 0.16])
    blue_rgb = np.array([0.12, 0.47, 0.71])

    # ---- row 0: particles
    sel_p = _sel_idx(len(particle_snaps), n_panels)
    pt_size = 2.5
    for col_idx, si in enumerate(sel_p):
        ax = fig.add_subplot(gs_left[0, col_idx])
        X1s = particle_snaps[si][0][0].cpu().numpy()
        X2s = particle_snaps[si][1][0].cpu().numpy()
        # Predicate-True (X1) -> red, matches rho=1 -> red convention of
        # the grid panels in row 2 so both rows share colour semantics.
        ax.scatter(X1s[:, 0], X1s[:, 1], c='#d62728', s=pt_size,
                   alpha=0.85, edgecolors='none', rasterized=True)
        ax.scatter(X2s[:, 0], X2s[:, 1], c='#1f77b4', s=pt_size,
                   alpha=0.85, edgecolors='none', rasterized=True)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'$t = {particle_times[si]:.1f}$', fontsize=14, pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        if col_idx == 0:
            ax.set_ylabel('Lagrangian', fontsize=12, color='black',
                          fontweight='bold')

    # ---- row 1: PDE grid densities
    sel_g = _sel_idx(len(pde_snaps), n_panels)
    for col_idx, si in enumerate(sel_g):
        ax = fig.add_subplot(gs_left[1, col_idx])
        rho = pde_snaps[si].cpu().numpy() if torch.is_tensor(pde_snaps[si]) else pde_snaps[si]
        a = np.clip(rho[..., None], 0.0, 1.0)
        rgb = a * red_rgb + (1 - a) * blue_rgb
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower',
                  extent=[0, 1, 0, 1], aspect='equal',
                  interpolation='bilinear')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        if col_idx == 0:
            ax.set_ylabel('Eulerian', fontsize=12, color='black',
                          fontweight='bold')

    # ---- right: mix-norm, two rows
    for row_idx, (color, t_arr, mix_arr, label_name) in enumerate([
        ('#1a66cc', np.asarray(particle_times), np.sqrt(np.maximum(np.asarray(particle_mix), 0.0)),
         'Particle'),
        ('#cc3311', np.asarray(pde_times),      np.sqrt(np.maximum(np.asarray(pde_mix), 0.0)),
         'PDE'),
    ]):
        ax = fig.add_subplot(gs_right[row_idx, 0])
        ax.set_box_aspect(1)
        ax.plot(t_arr, mix_arr, color=color, linewidth=2,
                label=r'$\Vert \rho-\bar\rho \Vert_{\dot H^{-1}}$')
        fit = _exp_fit(t_arr, mix_arr)
        if fit is not None:
            k_fit, b_fit = fit
            ax.plot(t_arr, k_fit * np.exp(b_fit * t_arr), 'k--',
                    linewidth=1.3,
                    label=fr'${k_fit:.2f}\,e^{{{b_fit:.2f}\,t}}$')
        ax.set_yscale('log')
        if row_idx == 1:
            ax.set_xlabel('$t$', fontsize=12)
        else:
            ax.tick_params(labelbottom=False)
            ax.set_title('mix-norm', fontsize=14, pad=4)
        ax.set_ylabel(r'$\Vert \theta \Vert_{\dot H^{-1}}$',
                      fontsize=10)
        ax.tick_params(which='both', labelsize=8)
        ax.margins(x=0)
        ax.legend(fontsize=7, loc='best')

    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(checkpoint, out_dir, N_x=256, N_T=None,
            N_grid_ctx=50, n_particles=6000, n_panels=6,
            param_mode='test', seed=0, family_override=None,
            device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    net, basis, ckpt_args = load_network(checkpoint, device)
    T       = ckpt_args.get('T', 5.0)
    M_basis = ckpt_args.get('M_basis', 2)
    N_T     = N_T if N_T is not None else ckpt_args.get('N_T', 20)
    K_mix   = ckpt_args.get('K_mix', 12)

    family_name = family_override or ckpt_args.get('family', 'rotation_line')
    level_set_K = ckpt_args.get('level_set_K', 3)
    family      = build_family(family_name, level_set_K=level_set_K)
    # Honour bounds/hole overrides stored in the checkpoint.
    ho = ckpt_args.get('hole_override', None)
    if ho:
        import json as _json
        family.hole = _json.loads(ho)
    bo = ckpt_args.get('bounds_override', None)
    if bo:
        import json as _json
        family.bounds = _json.loads(bo)

    torch.manual_seed(seed)
    # Balance-aware parameter draw: try N candidates and pick the one whose
    # {rho_1}-region area is closest to 1/2 (so theta = rho_1 - rho_2 has
    # mean near zero).
    g256 = (torch.arange(256, device=device, dtype=torch.float32) + 0.5) / 256
    _xx, _yy = torch.meshgrid(g256, g256, indexing='ij')
    _pts = torch.stack([_xx.reshape(-1), _yy.reshape(-1)], dim=-1)
    best_p, best_gap, best_frac = None, float('inf'), 0.0
    is_circle = getattr(family, 'name', '') == 'circle'
    for _k in range(60):
        cand = family.sample_param(mode=param_mode, device=device)
        if is_circle:
            cx, cy, r = float(cand[0]), float(cand[1]), float(cand[2])
            if r > min(cx, 1.0 - cx, cy, 1.0 - cy):
                continue
        frac = float(family.predicate(_pts, cand).float().mean())
        gap  = abs(frac - 0.5)
        if gap < best_gap:
            best_p, best_gap, best_frac = cand, gap, frac
        if gap <= 0.05:
            break
    p_vec = best_p
    print(f"Family: {family_name}  param = {family.describe(p_vec)}  "
          f"red_frac = {best_frac:.3f}")

    # PDE IC on high-res grid
    dtype = torch.float64
    rho0 = family_rho0_grid(family, p_vec, N_x, device=device, dtype=dtype)
    E1, E2 = _basis_on_faces(M_basis, N_x, device=device, dtype=dtype)

    # Network lambda schedule read from the context lattice used at training time
    X1_ctx, X2_ctx = family_lattice(family, p_vec, N_grid_ctx, device)
    lam_nn = network_lambda_schedule(net, X1_ctx, X2_ctx, T, N_T)

    # --- row 2: PDE rollout at high resolution
    print(f"PDE rollout: N_x={N_x}, N_T={N_T}")
    pde_traj, pde_times = simulate_eulerian_full(
        lam_nn.to(device=device, dtype=dtype),
        rho0.to(dtype=dtype), E1, E2, T)
    with torch.no_grad():
        pde_mix = np.asarray([
            float(grid_mix_norm_sq(r, K_max=K_mix, dye=True))
            for r in pde_traj])

    # --- row 1: particle rollout from a regular (cell-centered) mesh
    n_side = int(round(math.sqrt(n_particles)))
    print(f"Particle rollout: regular {n_side}x{n_side} mesh "
          f"({n_side * n_side} particles), N_T={N_T}")
    g = (torch.arange(n_side, device=device, dtype=torch.float32) + 0.5) / n_side
    xx, yy = torch.meshgrid(g, g, indexing='ij')
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    mask = family.predicate(pts, p_vec.to(device=device, dtype=pts.dtype))
    X1p = pts[mask].unsqueeze(0)
    X2p = pts[~mask].unsqueeze(0)

    # Context for the network is a separate lattice (already used for lam_nn)
    z = net.encode_context(X1_ctx, X2_ctx)

    part_snaps, part_times = particle_rollout(
        net, basis, z, X1p, X2p, T, N_T, N_snap=max(6, n_panels))

    # Particle mix-norm: empirical spectral dye cost, matches training
    dye = DyeMixCost(K_max=K_mix).to(device)
    with torch.no_grad():
        part_mix = [float(dye(s1, s2).item()) for s1, s2 in part_snaps]

    # --- figure
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"particle_pde_{family_name}.png")
    make_figure(part_snaps, part_times, part_mix,
                pde_traj,  pde_times,  pde_mix,
                out_path, n_panels=n_panels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=False)
    p.add_argument('--out_dir',    type=str,
                   default='out/eulerian_hires')
    p.add_argument('--results_dir', type=str, default='../results/2d',
                   help='Directory containing family checkpoint folders for --all.')
    p.add_argument('--N_x',        type=int, default=256)
    p.add_argument('--N_T',        type=int, default=None)
    p.add_argument('--N_grid_ctx', type=int, default=50)
    p.add_argument('--n_particles',type=int, default=6000)
    p.add_argument('--n_panels',   type=int, default=6)
    p.add_argument('--param_mode', type=str, default='test',
                   choices=['train', 'test'])
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--family',     type=str, default=None)
    p.add_argument('--all',        action='store_true',
                   help='Run all four 2D families (rotation_line, '
                        'general_line, circle, level_set).')
    args = p.parse_args()

    if args.all:
        families = [
            ('rotation_line', ('rotation_line_d1',)),
            ('general_line',  ('general_line_d2',)),
            ('circle',        ('circle_d3_wide', 'circle_d3')),
            ('level_set',     ('level_set_d6',)),
        ]
        checkpoints = []
        for fam_name, folders in families:
            candidates = [os.path.join(args.results_dir, folder, 'ckpt_final.pt')
                          for folder in folders]
            ckpt = next((path for path in candidates if os.path.isfile(path)), None)
            if ckpt is None:
                p.error(f"No checkpoint for {fam_name}; looked for: "
                        + ', '.join(candidates))
            checkpoints.append((fam_name, ckpt))
        for fam_name, ckpt in checkpoints:
            run_one(checkpoint=ckpt, out_dir=args.out_dir,
                    N_x=args.N_x, N_T=args.N_T,
                    N_grid_ctx=args.N_grid_ctx,
                    n_particles=args.n_particles,
                    n_panels=args.n_panels,
                    param_mode=args.param_mode, seed=args.seed,
                    family_override=fam_name)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint required (or use --all)")
        run_one(checkpoint=args.checkpoint, out_dir=args.out_dir,
                N_x=args.N_x, N_T=args.N_T,
                N_grid_ctx=args.N_grid_ctx,
                n_particles=args.n_particles,
                n_panels=args.n_panels,
                param_mode=args.param_mode, seed=args.seed,
                family_override=args.family)


if __name__ == '__main__':
    main()
