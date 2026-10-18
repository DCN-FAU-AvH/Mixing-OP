"""
test3d_cross_family.py -- Cross-family OOD visualisation (3D, Eulerian).

Loads ONE trained 3D checkpoint and evaluates it on ALL four partition
families. Produces a single figure with 4 rows:

    row 0 .. row 3   one per family in
                     (rotation_plane, general_plane, sphere, level_set_3d)

Each row shows 6 snapshots of theta(t, x) on a cube + the mix-norm curve.
The row whose family matches the checkpoint's training family is drawn
with blue spines (native / in-distribution); the other three rows use
red spines (out-of-distribution).

Usage
-----
    python -m example.test3d_cross_family \
        --checkpoint out/3d_rotation_plane_d2/ckpt_final.pt
"""

from __future__ import annotations

# Allow both module execution and direct execution of this example.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from model.basis3d import DivFreeBasis3D
from model     import MixingOperatorNet
from model.dataset3d import build_family_3d

from model.eulerian.pde_solver3d import (
    _basis_on_faces_3d,
    _assemble_velocity_3d,
    advect_interval_3d,
    family_rho0_grid_3d,
    grid_mix_norm_sq_3d,
)

from example.test3d_density import (
    network_lambda_schedule,
    simulate_with_trajectory,
    _theta_points,
    _setup_3d_axes,
    sample_partition,
)


FAMILY_ORDER = ('rotation_plane', 'general_plane', 'sphere', 'level_set_3d')


def _exp_fit(t: np.ndarray, m: np.ndarray):
    """Return (k, b) such that m ~= k * exp(b * t), fit on the tail."""
    idx = max(1, int(0.25 * len(t)))
    valid = m[idx:] > 0
    if valid.sum() < 2:
        return None, None
    slope, intercept = np.polyfit(t[idx:][valid],
                                  np.log(m[idx:][valid]), 1)
    return float(np.exp(intercept)), float(slope)


def main():
    p_ = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    p_.add_argument('--checkpoint',  type=str, required=True)
    p_.add_argument('--out_path',    type=str, default=None,
                    help='Output PNG path (default: next to the checkpoint, '
                         '"test3d_cross_<train_family>.png").')
    p_.add_argument('--level_set_K', type=int, default=None)
    p_.add_argument('--n_ctx',       type=int, default=1024)
    p_.add_argument('--T',           type=float, default=None)
    p_.add_argument('--N_T_sim',     type=int,   default=60)
    p_.add_argument('--N_x',         type=int,   default=48)
    p_.add_argument('--K_mix',       type=int,   default=None)
    p_.add_argument('--stride',      type=int,   default=3,
                    help='Scatter subsampling stride for the density panels')
    p_.add_argument('--s',           type=float, default=10.0,
                    help='Scatter marker size')
    p_.add_argument('--mode',        type=str,   default='test',
                    choices=['train', 'test'])
    p_.add_argument('--seed',        type=int,   default=0)
    p_.add_argument('--cfl',         type=float, default=0.5)
    p_.add_argument('--device',      type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p_.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    # ---- load ckpt + network --------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tr_args = ckpt['args']
    train_family_name = tr_args['family']
    level_set_K = args.level_set_K or tr_args.get('level_set_K', 3)
    T       = args.T      if args.T      is not None else tr_args['T']
    K_mix   = args.K_mix  if args.K_mix  is not None else tr_args.get('K_mix', 8)
    M_basis = tr_args['M_basis']

    basis = DivFreeBasis3D(M_basis).to(device)
    net = MixingOperatorNet(
        d=3, L=basis.L,
        h=tr_args['h'], n_heads=tr_args['n_heads'],
        n_layers_self=tr_args['n_layers_self'],
        n_layers_joint=tr_args['n_layers_joint'],
        t_emb_dim=tr_args['t_emb_dim'],
        dropout=tr_args.get('dropout', 0.0),
    ).to(device)
    net.load_state_dict(ckpt['model'])
    net.eval()

    print(f'Train family : {train_family_name}   (M_basis={M_basis}, L={basis.L})')

    # ---- shared face-basis (same M_basis, same N_x for all rows) --------
    E1, E2, E3 = _basis_on_faces_3d(M_basis, args.N_x,
                                    device=device, dtype=torch.float64)

    # ---- assemble per-family results ------------------------------------
    n_rows   = len(FAMILY_ORDER)
    n_panels = 6
    rows = []
    for fam_name in FAMILY_ORDER:
        print(f'--- {fam_name} ---')
        # Each family gets its own seed offset to avoid identical planes
        # between rotation_plane and general_plane.
        fam_seed = args.seed + hash(fam_name) % 97
        torch.manual_seed(fam_seed)

        family = build_family_3d(fam_name, level_set_K=level_set_K)
        p_vec, X1c, X2c, m1c, m2c = sample_partition(
            family, args.mode, args.n_ctx, device)
        rho0 = family_rho0_grid_3d(family, p_vec, args.N_x, device=device)

        lam = network_lambda_schedule(net, X1c, X2c, m1c, m2c,
                                      T=T, N_T=args.N_T_sim)
        lam_d = lam.to(device=device, dtype=torch.float64)
        rho_traj, times = simulate_with_trajectory(
            lam_d, rho0, E1, E2, E3, T=T, cfl=args.cfl)

        mix_curve = np.asarray([
            float(grid_mix_norm_sq_3d(r, K_max=K_mix, dye=True))
            for r in rho_traj])

        rows.append({
            'name':      fam_name,
            'is_native': (fam_name == train_family_name),
            'rho_traj':  rho_traj,
            'times':     times,
            'mix':       mix_curve,
            'param_str': family.describe(p_vec),
        })
        print(f'  mix^2: {mix_curve[0]:.3e} -> {mix_curve[-1]:.3e}   '
              f'ratio = {mix_curve[-1] / max(mix_curve[0], 1e-16):.3e}')

    # ---- figure layout (inch-based, parallel to 2D test_cross_family) --
    cell   = 2.1
    g_col  = 0.12
    g_row  = 0.18
    g_mid  = 1.0
    pad_l  = 0.05
    pad_r  = 0.25
    pad_t  = 0.55
    pad_b  = 0.40

    blk_w = n_panels * cell + (n_panels - 1) * g_col
    fig_w = pad_l + blk_w + g_mid + cell + pad_r
    fig_h = pad_b + n_rows * cell + (n_rows - 1) * g_row + pad_t

    fig = plt.figure(figsize=(fig_w, fig_h))

    L_left   = pad_l / fig_w
    R_left   = (pad_l + blk_w) / fig_w
    L_right  = (pad_l + blk_w + g_mid) / fig_w
    R_right  = (fig_w - pad_r) / fig_w
    B_       = pad_b / fig_h
    T_       = (fig_h - pad_t) / fig_h
    hs_frac  = g_row / cell
    ws_frac  = g_col / cell

    gs_left  = gridspec.GridSpec(
        n_rows, n_panels,
        left=L_left, right=R_left, bottom=B_, top=T_,
        hspace=hs_frac, wspace=ws_frac)
    gs_right = gridspec.GridSpec(
        n_rows, 1,
        left=L_right, right=R_right, bottom=B_, top=T_,
        hspace=hs_frac)

    for row_idx, row in enumerate(rows):
        rho_traj = row['rho_traj']
        times    = row['times']
        N_snap   = rho_traj.shape[0]
        sel = [int(round(i * (N_snap - 1) / (n_panels - 1)))
               for i in range(n_panels)]
        sel[0] = 0; sel[-1] = N_snap - 1

        row_color = '#1a66cc' if row['is_native'] else '#cc3311'

        for col_idx, si in enumerate(sel):
            ax = fig.add_subplot(gs_left[row_idx, col_idx], projection='3d')
            xyz, rgba = _theta_points(rho_traj[si].cpu().numpy(),
                                      stride=args.stride)
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                       c=rgba, s=args.s, depthshade=True, rasterized=True,
                       edgecolors='none', marker='o')
            _setup_3d_axes(ax)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
            if row_idx == 0:
                ax.set_title(f't = {float(times[si]):.1f}', fontsize=11, pad=2)
            if col_idx == 0:
                tag = ' (train)' if row['is_native'] else ' (test)'
                display_name = row['name'].replace('_', ' ')
                ax.text2D(-0.05, 0.5, display_name + tag,
                          transform=ax.transAxes, fontsize=10,
                          ha='right', va='center', rotation=90,
                          color=row_color)

        # mix-norm panel ----------------------------------------------------
        ax_m = fig.add_subplot(gs_right[row_idx, 0])
        ax_m.set_box_aspect(1)
        t_arr   = np.asarray(times)
        mix_arr = np.sqrt(np.clip(row['mix'], 0.0, None))
        ax_m.plot(t_arr, mix_arr, color=row_color, linewidth=2,
                  label=r'$\Vert \theta_h \Vert_{\dot{H}^{-1}(\Omega)}$')

        k_fit, b_fit = _exp_fit(t_arr, mix_arr)
        if k_fit is not None:
            ax_m.plot(t_arr, k_fit * np.exp(b_fit * t_arr), 'k--',
                      linewidth=1.2,
                      label=fr'${k_fit:.2f}\times e^{{{b_fit:.2f}\,t}}$')

        ax_m.set_yscale('log')
        ax_m.set_ylabel(r'$\Vert \theta_h \Vert_{\dot{H}^{-1}(\Omega)}$',
                        fontsize=10)
        if row_idx == n_rows - 1:
            ax_m.set_xlabel('t', fontsize=10)
        else:
            ax_m.tick_params(labelbottom=False)
        ax_m.tick_params(which='both', labelsize=8)
        ax_m.margins(x=0)
        ax_m.legend(fontsize=7, loc='best')
        if row_idx == 0:
            ax_m.set_title('mix-norm', fontsize=11, pad=2)

    # ---- save -----------------------------------------------------------
    out_path = args.out_path or os.path.join(
        os.path.dirname(args.checkpoint) or '.',
        f'test3d_cross_{train_family_name}.png')

    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
