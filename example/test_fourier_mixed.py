"""
test_fourier_mixed.py  --  Fourier-trained model evaluated on 4 different
partition settings, 2 experiments each (8 rows total):

  Rows 1-2 : rotation line (single line through the centre)
  Rows 3-4 : one circle
  Rows 5-6 : two circles (both circles = SAME density / same colour)
  Rows 7-8 : Fourier level set (native training distribution)

Layout is identical to test_multi.py (particle snapshots + square
log-scale mix-norm decay with exponential fit).

Usage
-----
  python -m example.test_fourier_mixed --checkpoint out/level_set_d6/ckpt_final.pt
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

import torch
import numpy as np
from scipy.stats import linregress

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.basis import DyeMixCost
from model.dataset import build_family
from model.util import make_lattice

from example.test_multi import load_model, integrate_with_snapshots


# ---------------------------------------------------------------
# Partition predicates
# ---------------------------------------------------------------

def rotation_line_mask(pts, theta):
    n = torch.tensor([math.cos(theta), math.sin(theta)],
                     dtype=pts.dtype, device=pts.device)
    return ((pts - 0.5) @ n) < 0.0


def one_circle_mask(pts, cx, cy, r):
    c = torch.tensor([cx, cy], dtype=pts.dtype, device=pts.device)
    return (pts - c).norm(dim=-1) < r


def two_circles_mask(pts, c1, r1, c2, r2):
    """Union of two discs forms ONE density (same colour in both)."""
    a = torch.tensor(c1, dtype=pts.dtype, device=pts.device)
    b = torch.tensor(c2, dtype=pts.dtype, device=pts.device)
    m1 = (pts - a).norm(dim=-1) < r1
    m2 = (pts - b).norm(dim=-1) < r2
    return m1 | m2


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Fourier-trained model on 4 partition settings, "
                    "2 experiments each.")
    p.add_argument('--checkpoint',  type=str,
                   default='out/level_set_d6/ckpt_final.pt')
    p.add_argument('--out_dir',     type=str, default='out/level_set_d6')
    p.add_argument('--N_grid',      type=int, default=50)
    p.add_argument('--N_snap',      type=int, default=200)
    p.add_argument('--K_mix',       type=int, default=24)
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--N_T',         type=int, default=200)
    p.add_argument('--level_set_K', type=int, default=3)
    p.add_argument('--device',      type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    net, basis, ckpt_args = load_model(args.checkpoint, device)

    T_ode   = ckpt_args.get('T', 5.0)
    N_T     = args.N_T or ckpt_args.get('N_T', 20)
    M_basis = ckpt_args.get('M_basis', 2)
    L       = basis.L
    print(f"Family(train)={ckpt_args.get('family')}  "
          f"M_basis={M_basis} L={L} T={T_ode} N_T={N_T}")

    dye_cost = DyeMixCost(args.K_mix).to(device)

    # Lattice
    lattice = make_lattice(args.N_grid, device=device)
    pts = lattice[0]

    # --- 8 experiments (4 settings x 2) ---
    experiments = []

    # 1-2  rotation line
    for i, theta in enumerate([math.pi / 4, math.pi / 2]):
        mask = rotation_line_mask(pts, theta)
        experiments.append(
            (f'rotation_line (theta={theta:.2f})', mask, 'ood'))

    # 3-4  one circle
    one_circle_params = [(0.5, 0.5, 0.25), (0.4, 0.6, 0.20)]
    for (cx, cy, r) in one_circle_params:
        mask = one_circle_mask(pts, cx, cy, r)
        experiments.append(
            (f'one_circle (c=({cx},{cy}), r={r})', mask, 'ood'))

    # 5-6  two circles (same colour)
    two_circle_params = [
        ((0.30, 0.30), 0.15, (0.70, 0.70), 0.15),
        ((0.25, 0.50), 0.13, (0.75, 0.50), 0.13),
    ]
    for (c1, r1, c2, r2) in two_circle_params:
        mask = two_circles_mask(pts, c1, r1, c2, r2)
        experiments.append(
            (f'two_circles ({c1},r={r1}; {c2},r={r2})', mask, 'ood'))

    # 7-8  level set (native)
    level_family = build_family('level_set', level_set_K=args.level_set_K)
    ho = ckpt_args.get('hole_override', None)
    if ho:
        import json as _json
        level_family.hole = _json.loads(ho)
    for _ in range(2):
        p_tensor = level_family.sample_param(mode='train', device=device)
        mask = level_family.predicate(pts, p_tensor)
        experiments.append(
            (f'level_set ({[round(float(v),2) for v in p_tensor]})',
             mask, 'native'))

    # --- Build cases (enforce non-empty both classes) ---
    cases = []
    for name, mask, tag in experiments:
        X1 = pts[mask].unsqueeze(0)
        X2 = pts[~mask].unsqueeze(0)
        if X1.shape[1] == 0 or X2.shape[1] == 0:
            raise RuntimeError(f"Degenerate partition: {name}")
        cases.append({'name': name, 'X1': X1, 'X2': X2, 'tag': tag})

    n_rows   = len(cases)
    n_panels = 6
    pt_size  = max(1.0, 600 / args.N_grid)

    # =========================================================
    # Figure layout (identical to test_multi.py)
    # =========================================================
    cell   = 1.8
    g_col  = 0.08
    g_row  = 0.08
    g_mid  = 1.2
    pad_l  = 0.05
    pad_r  = 0.20
    pad_t  = 0.35
    pad_b  = 0.35

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

    gs_left  = gridspec.GridSpec(
        n_rows, n_panels,
        left=L_left, right=R_left, bottom=gs_B, top=gs_T,
        hspace=hs_frac, wspace=ws_frac)
    gs_right = gridspec.GridSpec(
        n_rows, 1,
        left=L_right, right=R_right, bottom=gs_B, top=gs_T,
        hspace=hs_frac)

    for row, case in enumerate(cases):
        name = case['name']; tag = case['tag']
        print(f"\n--- [{tag}] {name} ---")

        X1, X2 = case['X1'], case['X2']
        z = net.encode_context(X1, X2)

        snapshots, snap_times = integrate_with_snapshots(
            net, basis, z, X1, X2, T_ode, N_T, N_snap=args.N_snap)
        norms = [dye_cost(s1, s2).item() for s1, s2 in snapshots]
        ratio = norms[-1] / norms[0] if norms[0] > 0 else 0.0

        if n_panels >= len(snapshots):
            sel = list(range(len(snapshots)))
        else:
            sel = [int(round(i * (len(snapshots) - 1) / (n_panels - 1)))
                   for i in range(n_panels)]
            sel[0] = 0; sel[-1] = len(snapshots) - 1

        row_color = '#1a66cc' if tag == 'native' else '#cc3311'

        for col_idx, si in enumerate(sel):
            ax = fig.add_subplot(gs_left[row, col_idx])
            X1s = snapshots[si][0][0].cpu().numpy()
            X2s = snapshots[si][1][0].cpu().numpy()
            ax.scatter(X1s[:, 0], X1s[:, 1], c='#1f77b4', s=pt_size,
                       alpha=0.8, edgecolors='none', rasterized=True)
            ax.scatter(X2s[:, 0], X2s[:, 1], c='#d62728', s=pt_size,
                       alpha=0.8, edgecolors='none', rasterized=True)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f't = {snap_times[si]:.1f}', fontsize=14, pad=4)
            for spine in ax.spines.values():
                spine.set_edgecolor(row_color)
                spine.set_linewidth(1.5)

        ax_norm = fig.add_subplot(gs_right[row, 0])
        ax_norm.set_box_aspect(1)
        t_arr   = np.array(snap_times)
        mix_arr = np.sqrt(np.array(norms))
        ax_norm.plot(t_arr, mix_arr, color=row_color, linewidth=2,
                     label=r'$\Vert \theta_h \Vert_{\dot{H}^{-1}}$')
        idx = max(1, int(0.25 * len(t_arr)))
        valid = mix_arr[idx:] > 0
        if valid.sum() >= 2:
            t_fit  = t_arr[idx:][valid]
            log_mx = np.log(mix_arr[idx:][valid])
            slope, intercept, *_ = linregress(t_fit, log_mx)
            k_fit = float(np.exp(intercept)); b_fit = float(slope)
            ax_norm.plot(t_arr, k_fit * np.exp(b_fit * t_arr), 'k--',
                         linewidth=1.5,
                         label=fr'${k_fit:.2f}\times e^{{{b_fit:.2f}\,t}}$')
        ax_norm.set_yscale('log')
        if row == n_rows - 1:
            ax_norm.set_xlabel('t', fontsize=11)
        else:
            ax_norm.tick_params(labelbottom=False)
        ax_norm.set_ylabel(r'$\Vert \theta_h \Vert_{\dot{H}^{-1}}$',
                           fontsize=10)
        ax_norm.tick_params(which='both', labelsize=8)
        ax_norm.margins(x=0)
        ax_norm.legend(fontsize=7, loc='best')
        if row == 0:
            ax_norm.set_title('mix-norm', fontsize=14, pad=4)

        print(f"  Norm: {norms[0]:.6f} -> {norms[-1]:.6f} "
              f"({(1 - ratio) * 100:.1f}% reduction)")

    save_path = os.path.join(args.out_dir, 'test_fourier_mixed.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
