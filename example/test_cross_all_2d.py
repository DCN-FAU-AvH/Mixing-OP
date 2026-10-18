"""
test_cross_all_2d.py  --  Cross-family 2D visualisation: native + 3 OOD.

Given a model trained on one 2D family, evaluates it on all four 2D families
(native + the three others). Produces a 4-row figure:

  Row 0       : native family (blue spine)
  Rows 1..3   : the three other families (red spines)

Layout matches the 3D cross-family figures (particle snapshots + log-scale
mix-norm decay on the right).

Usage
-----
  python -m example.test_cross_all_2d --checkpoint path/to/ckpt_final.pt
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
from example.test_multi import (
    load_model,
    integrate_with_snapshots,
    sample_vis_params,
    make_cases,
)


ALL_FAMILIES_2D = ['rotation_line', 'general_line', 'circle', 'level_set']


def _one_param(family, device, seed, mode='train'):
    """Draw a single parameter point from `family` in the requested mode."""
    params, is_train, _ = sample_vis_params(
        family, n_train=(1 if mode == 'train' else 0),
        n_test=(0 if mode == 'train' else 1),
        device=device, seed=seed)
    return params[0], is_train[0]


def _balanced_param(family, device, seed, mode='train',
                    n_attempts=60, tol=0.05):
    """
    Draw a parameter whose partition has red/blue area closest to 50/50.
    Tries `n_attempts` random draws (different seed offsets), returns the
    first draw within `tol` of 0.5 or else the best-balanced draw seen.
    For the circle family, params whose disk escapes [0,1]^2 are rejected.
    """
    g = (torch.arange(256, device=device, dtype=torch.float32) + 0.5) / 256
    xx, yy = torch.meshgrid(g, g, indexing='ij')
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    is_circle = getattr(family, 'name', '') == 'circle'
    best, best_gap = None, float('inf')
    best_frac, best_it = 0.0, None
    for k in range(n_attempts):
        p, it = _one_param(family, device, seed + 10_000 * k, mode=mode)
        if is_circle:
            cx, cy, r = float(p[0]), float(p[1]), float(p[2])
            if r > min(cx, 1.0 - cx, cy, 1.0 - cy):
                continue
        frac = float(family.predicate(pts, p).float().mean())
        gap  = abs(frac - 0.5)
        if gap < best_gap:
            best, best_gap, best_frac, best_it = p, gap, frac, it
        if gap <= tol:
            return p, it, frac
    return best, best_it, best_frac


def main():
    p = argparse.ArgumentParser(
        description="2D cross-family visualisation: 1 native row + 3 OOD rows.")
    p.add_argument('--checkpoint',  type=str, required=True)
    p.add_argument('--out_dir',     type=str, default='out')
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

    train_family_name = ckpt_args.get('family', 'rotation_line')
    train_level_set_K = ckpt_args.get('level_set_K', 3)

    # Canonical row order, identical across all four figures. The native row
    # is whichever family the checkpoint was trained on; the remaining three
    # rows are OOD.
    ordered = list(ALL_FAMILIES_2D)

    # Build one case per family.
    cases = []
    for i, fam_name in enumerate(ordered):
        lsK = train_level_set_K if fam_name == train_family_name \
              else args.level_set_K
        fam = build_family(fam_name, level_set_K=lsK)
        is_native = (fam_name == train_family_name)
        if is_native:
            ho = ckpt_args.get('hole_override', None)
            if ho:
                import json as _json
                fam.hole = _json.loads(ho)
            bo = ckpt_args.get('bounds_override', None)
            if bo:
                import json as _json
                fam.bounds = _json.loads(bo)
        # For the circle family, always use the wider-radius bounds so the
        # partition can reach ~50/50 area (mean-zero theta).
        if fam_name == 'circle':
            fam.bounds = [[0.25, 0.75], [0.25, 0.75], [0.15, 0.45]]
            if not is_native:
                fam.hole = [[0.40, 0.60], [0.40, 0.60], [0.30, 0.40]]
        param, _, frac = _balanced_param(
            fam, device, args.seed + i, mode='train')
        print(f"  param={fam.describe(param)}  red_frac={frac:.3f}")
        c = make_cases(fam, [param], [is_native], args.N_grid, device=device)
        if not c:
            raise RuntimeError(
                f"Could not build a case for family '{fam_name}'.")
        cases.append(c[0])

    n_rows   = len(cases)
    n_panels = 6
    pt_size  = max(1.0, 600 / args.N_grid)

    native_idx = ordered.index(train_family_name)
    print(f"Train family = {train_family_name}  (native row = {native_idx})")
    print(f"Row order   = {ordered}")
    print(f"M_basis={M_basis}  L={L}  T={T_ode}  N_T={N_T}")

    dye_cost = DyeMixCost(args.K_mix).to(device)

    # =========================================================
    # Layout (inch-based, identical to test_multi.py)
    # =========================================================
    cell   = 1.8
    g_col  = 0.08
    g_row  = 0.08
    g_mid  = 1.2
    pad_l  = 0.95   # room for the left-side family labels
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
        name      = case['name']
        is_native = case['is_train']
        tag       = 'native' if is_native else 'OOD'
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

        fam_name = ordered[row]

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
            if col_idx == 0:
                tag = ' (train)' if is_native else ' (test)'
                display_name = fam_name.replace('_', ' ')
                ax.text(-0.12, 0.5, display_name + tag,
                        transform=ax.transAxes, fontsize=11,
                        ha='right', va='center', rotation=90,
                        color=row_color)
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.2)

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
        args.out_dir, f"cross_all_{train_family_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
