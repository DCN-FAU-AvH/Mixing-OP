"""
test_multi.py  --  Family-aware generalization visualisation.

Loads the partition family from the checkpoint, draws N_train_vis parameter
points from OUTSIDE the test hole and N_test_vis points from INSIDE the
hole, rolls out the trained operator on each, and renders:

  1. test_multi_{family}.png           particle snapshots + log-scale decay
  2. test_lambda_compare_{family}.png  lambda(t) per parameter point
  3. test_generalization_{family}.png  scatter of final mix-norm reduction
                                       vs a scalar parameter axis
                                       (train=blue, test=red)

Usage
-----
  python -m example.test_multi --checkpoint out/rotation_line_d1/ckpt_final.pt
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

from model.basis import DivFreeBasis, DyeMixCost
from model   import MixingOperatorNet
from train   import _lam_at
from model.dataset import build_family
from model.util import make_lattice


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = 'cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    a = ckpt.get('args', {})
    basis = DivFreeBasis(a.get('M_basis', 2)).to(device)
    net = MixingOperatorNet(
        d=2, L=basis.L, h=a.get('h', 256),
        n_heads=a.get('n_heads', 4),
        n_layers_self=a.get('n_layers_self', 1),
        n_layers_joint=a.get('n_layers_joint', 1),
        t_emb_dim=a.get('t_emb_dim', 64),
    ).to(device)
    net.load_state_dict(ckpt['model'])
    net.eval()
    return net, basis, a


@torch.no_grad()
def integrate_with_snapshots(net, basis, z, X1, X2, T, N_T, N_snap=20):
    dt = T / N_T
    snap_interval = max(1, N_T // N_snap)
    snapshots = [(X1.clone(), X2.clone())]
    snap_times = [0.0]

    for s in range(N_T):
        t_s = s * dt

        lam1 = _lam_at(net, z, t_s, T)
        k1_1 = basis.velocity(X1, lam1)
        k1_2 = basis.velocity(X2, lam1)

        lam2 = _lam_at(net, z, t_s + 0.5 * dt, T)
        k2_1 = basis.velocity(X1 + 0.5 * dt * k1_1, lam2)
        k2_2 = basis.velocity(X2 + 0.5 * dt * k1_2, lam2)

        k3_1 = basis.velocity(X1 + 0.5 * dt * k2_1, lam2)
        k3_2 = basis.velocity(X2 + 0.5 * dt * k2_2, lam2)

        lam4 = _lam_at(net, z, t_s + dt, T)
        k4_1 = basis.velocity(X1 + dt * k3_1, lam4)
        k4_2 = basis.velocity(X2 + dt * k3_2, lam4)

        X1 = (X1 + (dt / 6) * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)).clamp(0, 1)
        X2 = (X2 + (dt / 6) * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)).clamp(0, 1)

        if (s + 1) % snap_interval == 0 or (s + 1) == N_T:
            snapshots.append((X1.clone(), X2.clone()))
            snap_times.append((s + 1) * dt)

    return snapshots, snap_times


@torch.no_grad()
def evaluate_lambdas_dense(net, z, T, n_points=200):
    times = np.linspace(0, T, n_points)
    lam_all = []
    for t_val in times:
        lam = _lam_at(net, z, float(t_val), T)
        lam_all.append(lam[0].cpu().numpy())
    return times, np.stack(lam_all, axis=0)


# ---------------------------------------------------------------
# Parameter sampling for visualisation
# ---------------------------------------------------------------

def _train_points_1d(lo, hi, hole, n):
    """Evenly spaced points in [lo, hi] \\ hole."""
    hlo, hhi = hole
    left_w  = max(0.0, hlo - lo)
    right_w = max(0.0, hi - hhi)
    total_w = left_w + right_w
    if total_w <= 0:
        return np.linspace(lo, hi, n)
    n_left  = max(1, int(round(n * left_w / total_w))) if left_w > 0 else 0
    n_right = n - n_left
    pts = []
    if n_left > 0:
        pts.extend(np.linspace(lo, hlo, n_left + 2)[1:-1] if n_left > 1
                   else [0.5 * (lo + hlo)])
    if n_right > 0:
        pts.extend(np.linspace(hhi, hi, n_right + 2)[1:-1] if n_right > 1
                   else [0.5 * (hhi + hi)])
    return np.array(pts[:n])


def _test_points_1d(hole, n):
    hlo, hhi = hole
    if n == 1:
        return np.array([0.5 * (hlo + hhi)])
    return np.linspace(hlo, hhi, n + 2)[1:-1]


def sample_vis_params(family, n_train, n_test, device, seed=0):
    """
    Return (params, is_train, axis_vals)
      params     : list of length (n_train + n_test) of (dim,) tensors
      is_train   : list of bool
      axis_vals  : list of scalar floats used as the x-axis in the
                   generalization scatter plot. For dim==1 we use the
                   parameter itself; otherwise we project on dim 0.
    """
    params, is_train, axis = [], [], []

    # 1D: evenly spaced points on the complement / inside hole
    if family.dim == 1 and family.hole[0] is not None:
        lo, hi = family.bounds[0]
        hole   = family.hole[0]
        tr = _train_points_1d(lo, hi, hole, n_train)
        te = _test_points_1d(hole, n_test)
        for v in tr:
            params.append(torch.tensor([float(v)], device=device))
            is_train.append(True); axis.append(float(v))
        for v in te:
            params.append(torch.tensor([float(v)], device=device))
            is_train.append(False); axis.append(float(v))
        return params, is_train, axis

    # higher-dim: seeded rejection sampler
    g = torch.Generator(device='cpu').manual_seed(seed)

    def _draw(mode, k):
        # Swap in a temporary RNG for reproducibility
        saved = torch.random.get_rng_state()
        torch.random.set_rng_state(g.get_state())
        out = []
        for _ in range(k):
            p = family.sample_param(mode=mode, device=device)
            out.append(p)
        g.set_state(torch.random.get_rng_state())
        torch.random.set_rng_state(saved)
        return out

    tr = _draw('train', n_train)
    te = _draw('test',  n_test)
    for p in tr:
        params.append(p); is_train.append(True)
        axis.append(float(p[0]))
    for p in te:
        params.append(p); is_train.append(False)
        axis.append(float(p[0]))
    return params, is_train, axis


# ---------------------------------------------------------------
# Test cases from parameters
# ---------------------------------------------------------------

def make_cases(family, params, is_train, N_grid, device='cpu'):
    lattice = make_lattice(N_grid, device=device)  # (1, N^2, 2)
    pts = lattice[0]
    cases = []
    for p, tr in zip(params, is_train):
        mask = family.predicate(pts, p)             # True = rho_1
        X1 = pts[mask].unsqueeze(0)
        X2 = pts[~mask].unsqueeze(0)
        if X1.shape[1] == 0 or X2.shape[1] == 0:
            continue
        cases.append({
            'name':    family.describe(p),
            'param':   p,
            'is_train': tr,
            'X1':      X1,
            'X2':      X2,
        })
    return cases


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Family-aware two-density operator generalization demo")
    p.add_argument('--checkpoint',   type=str, default='out/ckpt_final.pt')
    p.add_argument('--out_dir',      type=str, default='out')
    p.add_argument('--N_grid',       type=int, default=50)
    p.add_argument('--N_snap',       type=int, default=200)
    p.add_argument('--K_mix',        type=int, default=24)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--N_T',          type=int, default=200)
    p.add_argument('--n_train_vis',  type=int, default=1)
    p.add_argument('--n_test_vis',   type=int, default=3)
    p.add_argument('--device',       type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    net, basis, ckpt_args = load_model(args.checkpoint, device)

    T       = ckpt_args.get('T', 5.0)
    N_T     = args.N_T or ckpt_args.get('N_T', 20)
    M_basis = ckpt_args.get('M_basis', 2)
    L       = basis.L

    # --- Rebuild the partition family from the checkpoint args ---
    family_name = ckpt_args.get('family', 'rotation_line')
    level_set_K = ckpt_args.get('level_set_K', 3)
    family = build_family(family_name, level_set_K=level_set_K)
    # honour any hole_override the run was started with
    ho = ckpt_args.get('hole_override', None)
    if ho:
        import json as _json
        family.hole = _json.loads(ho)
    print(f"Family={family.name}  dim={family.dim}  hole={family.hole}")
    print(f"M_basis={M_basis}  L={L}  T={T}  N_T={N_T}")

    dye_cost = DyeMixCost(args.K_mix).to(device)

    # --- Parameter points: train (outside hole) + test (inside hole) ---
    params, is_train, axis_vals = sample_vis_params(
        family, args.n_train_vis, args.n_test_vis, device, seed=args.seed)
    cases = make_cases(family, params, is_train, args.N_grid, device=device)
    n_rows = len(cases)
    if n_rows == 0:
        raise RuntimeError("No valid visualization cases produced.")

    n_panels = 6
    pt_size  = max(1.0, 600 / args.N_grid)

    # =========================================================
    # Figure 1: particle evolution + log-scale decay per row
    # =========================================================
    # Layout computed in inches so gaps are uniform and cells
    # are truly square regardless of n_rows.
    cell   = 1.8        # size of each square cell (inches)
    g_col  = 0.08       # gap between particle columns (inches)
    g_row  = 0.08       # gap between rows (inches)
    g_mid  = 1.2        # gap between particle block and mix-norm (inches)
    pad_l  = 0.05       # left margin
    pad_r  = 0.20       # right margin
    pad_t  = 0.35       # top margin  (room for titles)
    pad_b  = 0.35       # bottom margin (room for x-label on last row)

    blk_w  = n_panels * cell + (n_panels - 1) * g_col   # particle block width
    fig_w  = pad_l + blk_w + g_mid + cell + pad_r
    fig_h  = pad_b + n_rows * cell + (n_rows - 1) * g_row + pad_t

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Convert inch measurements to figure fractions
    L_left   = pad_l / fig_w
    R_left   = (pad_l + blk_w) / fig_w
    L_right  = (pad_l + blk_w + g_mid) / fig_w
    R_right  = (fig_w - pad_r) / fig_w
    gs_B     = pad_b / fig_h
    gs_T     = (fig_h - pad_t) / fig_h
    hs_frac  = g_row / cell    # hspace as fraction of cell height
    ws_frac  = g_col / cell    # wspace as fraction of cell width

    gs_left = gridspec.GridSpec(
        n_rows, n_panels,
        left=L_left, right=R_left, bottom=gs_B, top=gs_T,
        hspace=hs_frac, wspace=ws_frac)

    gs_right = gridspec.GridSpec(
        n_rows, 1,
        left=L_right, right=R_right, bottom=gs_B, top=gs_T,
        hspace=hs_frac)

    all_dense_lambdas = []
    all_dense_times   = None
    density_names     = []
    reduction_fracs   = []   # fractional mix-norm reduction for Figure 3
    row_is_train      = []

    for row, case in enumerate(cases):
        name = case['name']
        density_names.append(name)
        row_is_train.append(case['is_train'])
        tag  = 'train' if case['is_train'] else 'test '
        print(f"\n--- [{tag}] {name} ---")

        X1, X2 = case['X1'], case['X2']
        z = net.encode_context(X1, X2)

        snapshots, snap_times = integrate_with_snapshots(
            net, basis, z, X1, X2, T, N_T, N_snap=args.N_snap)
        norms = [dye_cost(s1, s2).item() for s1, s2 in snapshots]

        ratio = norms[-1] / norms[0] if norms[0] > 0 else 0.0
        reduction_fracs.append(1.0 - ratio)

        dense_times, dense_lambdas = evaluate_lambdas_dense(net, z, T)
        all_dense_lambdas.append(dense_lambdas)
        all_dense_times = dense_times

        n_snaps = len(snapshots)
        if n_snaps <= n_panels:
            sel = list(range(n_snaps))
        else:
            sel = [int(round(i * (n_snaps - 1) / (n_panels - 1)))
                   for i in range(n_panels)]
            sel[0] = 0; sel[-1] = n_snaps - 1

        row_color = '#1a66cc' if case['is_train'] else '#cc3311'

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

        # --- log-scale decay + exponential fit ---
        ax_norm = fig.add_subplot(gs_right[row, 0])
        ax_norm.set_box_aspect(1)       # force square shape
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

    save_path = os.path.join(args.out_dir, f"test_multi_{family.name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {save_path}")
    plt.close()

    # =========================================================
    # Figure 2: lambda comparison
    # =========================================================
    ms_np = basis.ms.cpu().numpy().astype(int)
    ns_np = basis.ns.cpu().numpy().astype(int)
    cmap  = plt.cm.tab20 if L > 10 else plt.cm.tab10

    fig2, axes = plt.subplots(1, n_rows, figsize=(4.5 * n_rows, 3.5))
    if n_rows == 1:
        axes = [axes]

    for row in range(n_rows):
        ax = axes[row]
        for l_idx in range(L):
            c = cmap(l_idx % 20)
            label = (f'$\\lambda_{{{ms_np[l_idx]},{ns_np[l_idx]}}}$'
                     if L <= 12 else None)
            ax.plot(all_dense_times, all_dense_lambdas[row][:, l_idx],
                    color=c, linewidth=1.2, alpha=0.8, label=label)
        tag = 'train' if row_is_train[row] else 'test'
        col = '#1a66cc' if row_is_train[row] else '#cc3311'
        ax.set_title(f'[{tag}] {density_names[row]}', fontsize=9,
                     fontweight='bold', color=col)
        ax.set_xlabel('t', fontsize=9)
        ax.set_ylabel(r'$\lambda_\ell(t)$', fontsize=9)
        if L <= 12:
            ax.legend(fontsize=6, ncol=2, loc='best')

    fig2.suptitle(
        f'$\\lambda(t)$  family={family.name}  (M={M_basis}, L={L})',
        fontsize=12, fontweight='bold')
    fig2.tight_layout()
    save_path2 = os.path.join(args.out_dir,
                              f"test_lambda_compare_{family.name}.png")
    fig2.savefig(save_path2, dpi=200, bbox_inches='tight')
    print(f"Saved lambda comparison to {save_path2}")
    plt.close()

    # =========================================================
    # Figure 3: generalization scatter
    # =========================================================
    fig3, ax3 = plt.subplots(figsize=(6, 4.2))
    xs_tr = [axis_vals[i] for i in range(len(cases)) if row_is_train[i]]
    ys_tr = [reduction_fracs[i] for i in range(len(cases)) if row_is_train[i]]
    xs_te = [axis_vals[i] for i in range(len(cases)) if not row_is_train[i]]
    ys_te = [reduction_fracs[i] for i in range(len(cases)) if not row_is_train[i]]

    ax3.scatter(xs_tr, ys_tr, c='#1a66cc', s=80, label='train (outside hole)',
                edgecolors='k', linewidths=0.5, zorder=3)
    ax3.scatter(xs_te, ys_te, c='#cc3311', s=80, label='test (inside hole)',
                edgecolors='k', linewidths=0.5, marker='D', zorder=3)

    # Shade the hole interval when dim==1
    if family.dim == 1 and family.hole[0] is not None:
        hlo, hhi = family.hole[0]
        ax3.axvspan(hlo, hhi, color='#cc3311', alpha=0.08,
                    label='test hole')

    axis_label = (r'$\theta$' if family.name == 'rotation_line'
                  else f'param[0] ({family.name})')
    ax3.set_xlabel(axis_label, fontsize=11)
    ax3.set_ylabel(r'mix-norm reduction $1 - \|f_T\|^2 / \|f_0\|^2$',
                   fontsize=11)
    ax3.set_title(
        f'Generalization: family={family.name} (dim={family.dim})',
        fontsize=12, fontweight='bold')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=9, loc='best')
    fig3.tight_layout()
    save_path3 = os.path.join(args.out_dir,
                              f"test_generalization_{family.name}.png")
    fig3.savefig(save_path3, dpi=200, bbox_inches='tight')
    print(f"Saved generalization plot to {save_path3}")
    plt.close()


if __name__ == '__main__':
    main()
