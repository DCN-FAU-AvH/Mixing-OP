"""
test_cross_family.py  --  Cross-family OOD generalization visualisation.

Loads a trained model (trained on one partition family, e.g. circle), and
evaluates it on TWO partition families:

  Rows 0..3  :  the model's NATIVE family (train distribution, outside hole)
  Rows 4..7  :  a DIFFERENT family (fully out-of-distribution)

Produces test_cross_{train_family}_to_{eval_family}.png with the same
layout as example/test_multi.py (particle snapshots + log-scale mix-norm decay).

Usage
-----
  python -m example.test_cross_family --checkpoint out/circle_d3/ckpt_final.pt \
                              --eval_family level_set
"""

from __future__ import annotations

# Allow both module execution and direct execution of this example.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
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

from example.test_multi import (
    load_model,
    integrate_with_snapshots,
    sample_vis_params,
    make_cases,
)


def _sample_native_train(family, n, device, seed):
    """Sample n parameter points from the TRAIN region (outside the hole)."""
    params, is_train, _ = sample_vis_params(
        family, n_train=n, n_test=0, device=device, seed=seed)
    return params, is_train


def _sample_ood(family, n, device, seed):
    """Sample n parameter points from the full parameter space of a DIFFERENT
    family. We use train-mode samples so they are in-range and valid."""
    params, is_train, _ = sample_vis_params(
        family, n_train=n, n_test=0, device=device, seed=seed + 1000)
    # mark as "test" (False) so row borders / labels differ
    return params, [False] * len(params)


def main():
    p = argparse.ArgumentParser(
        description="Cross-family OOD generalization visualisation")
    p.add_argument('--checkpoint',  type=str, required=True,
                   help='Path to trained model checkpoint.')
    p.add_argument('--eval_family', type=str, required=True,
                   choices=['rotation_line', 'general_line',
                            'circle', 'level_set'],
                   help='OOD family to evaluate the trained model on.')
    p.add_argument('--out_dir',     type=str, default='out')
    p.add_argument('--N_grid',      type=int, default=50)
    p.add_argument('--N_snap',      type=int, default=200)
    p.add_argument('--K_mix',       type=int, default=24)
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--N_T',         type=int, default=200)
    p.add_argument('--n_rows_each', type=int, default=4,
                   help='Number of rows for each of train / OOD blocks.')
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

    # --- Build the two families ---
    train_family_name = ckpt_args.get('family', 'rotation_line')
    train_level_set_K = ckpt_args.get('level_set_K', 3)
    train_family = build_family(train_family_name,
                                level_set_K=train_level_set_K)
    ho = ckpt_args.get('hole_override', None)
    if ho:
        import json as _json
        train_family.hole = _json.loads(ho)

    eval_family = build_family(args.eval_family,
                               level_set_K=args.level_set_K)

    print(f"Train family = {train_family.name}  (dim={train_family.dim})")
    print(f"Eval  family = {eval_family.name}   (dim={eval_family.dim})")
    print(f"M_basis={M_basis}  L={L}  T={T_ode}  N_T={N_T}")

    if train_family.name == eval_family.name:
        raise ValueError(
            f"--eval_family must differ from the training family "
            f"('{train_family.name}'). Use example/test_multi.py for "
            f"in-family evaluation.")

    dye_cost = DyeMixCost(args.K_mix).to(device)

    # --- Assemble 4 train-family + 4 eval-family cases ---
    n = args.n_rows_each
    tr_params, tr_is_train = _sample_native_train(
        train_family, n, device, args.seed)
    te_params, te_is_train = _sample_ood(
        eval_family, n, device, args.seed)

    tr_cases = make_cases(train_family, tr_params, tr_is_train,
                          args.N_grid, device=device)
    te_cases = make_cases(eval_family,  te_params, te_is_train,
                          args.N_grid, device=device)
    cases = tr_cases + te_cases
    n_rows = len(cases)
    if n_rows == 0:
        raise RuntimeError("No valid visualization cases produced.")

    n_panels = 6
    pt_size  = max(1.0, 600 / args.N_grid)

    # =========================================================
    # Figure layout (inch-based, identical to example/test_multi.py)
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
        name = case['name']
        is_native = case['is_train']      # True: train family, False: OOD
        tag = 'train' if is_native else 'OOD'
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

        row_color = '#1a66cc' if is_native else '#cc3311'

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

    save_path = os.path.join(
        args.out_dir,
        f"test_cross_{train_family.name}_to_{eval_family.name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
