"""
test3d.py  --  Visualise a trained 3D MixingOperatorNet.

For a chosen partition family and checkpoint, this script:

  1. Draws a random test-mode partition (inside the held-out hole).
  2. Encodes the context, rolls RK4 through N_T_draw time-steps, and
     records particle positions + lambda(t) + dye mix-norm at each step.
  3. Produces four figures:
       <stem>_scatter3d.png  -- 3D scatter at 6 time snapshots
       <stem>_slices.png     -- axis-aligned slab slices (3 rows x 5 cols)
       <stem>_lambda.png     -- lambda_l(t) coefficients
       <stem>_mix_norm.png   -- dye mix-norm decay (2D-style, square)

Usage
-----
  python -m example.test3d --checkpoint out/3d_rotation_plane_d2/ckpt_final.pt
"""

from __future__ import annotations

# Allow both module execution and direct execution of this example.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import argparse
import math

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3d proj)

from model.basis3d import DivFreeBasis3D, DyeMixCost3D
from model     import MixingOperatorNet
from model.dataset3d import build_family_3d, make_partition_sampler_3d


# ---------------------------------------------------------------------------
# Rollout helper: store positions and lambda at every sub-step
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout(net, basis, dye_cost,
            X1_0, X2_0, m1, m2, T, N_T):
    """
    RK4 roll-out that records (X1, X2, lambda, mix_norm) at each step.
    Returns
    -------
    ts       : (N_T+1,)
    X1_hist  : (N_T+1, B, n1, 3)
    X2_hist  : (N_T+1, B, n2, 3)
    lam_hist : (N_T+1, B, L)   lambda(t_k)
    mix_hist : (N_T+1, B)      dye mix-norm at each step
    """
    dt = T / N_T
    B  = X1_0.shape[0]
    X1 = X1_0.clone()
    X2 = X2_0.clone()

    X1_hist = [X1.clone()]
    X2_hist = [X2.clone()]
    lam_hist = []
    mix_hist = [dye_cost(X1, X2, m1, m2).cpu()]

    z = net.encode_context(X1, X2, m1, m2)

    for s in range(N_T):
        t_s = s * dt

        t_norm = z.new_full((B,), t_s / T)
        lam1 = net.forward_from_ctx(z, t_norm)
        lam_hist.append(lam1.cpu())

        k1_1 = basis.velocity(X1, lam1)
        k1_2 = basis.velocity(X2, lam1)

        t_norm = z.new_full((B,), (t_s + 0.5 * dt) / T)
        lam2 = net.forward_from_ctx(z, t_norm)
        k2_1 = basis.velocity(X1 + 0.5 * dt * k1_1, lam2)
        k2_2 = basis.velocity(X2 + 0.5 * dt * k1_2, lam2)

        k3_1 = basis.velocity(X1 + 0.5 * dt * k2_1, lam2)
        k3_2 = basis.velocity(X2 + 0.5 * dt * k2_2, lam2)

        t_norm = z.new_full((B,), (t_s + dt) / T)
        lam4 = net.forward_from_ctx(z, t_norm)
        k4_1 = basis.velocity(X1 + dt * k3_1, lam4)
        k4_2 = basis.velocity(X2 + dt * k3_2, lam4)

        v1 = (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6.0
        v2 = (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6.0
        X1 = (X1 + dt * v1).clamp(0.0, 1.0)
        X2 = (X2 + dt * v2).clamp(0.0, 1.0)

        X1_hist.append(X1.clone())
        X2_hist.append(X2.clone())
        mix_hist.append(dye_cost(X1, X2, m1, m2).cpu())

    # Final lambda at t = T (evaluated at endpoint for completeness).
    t_norm = z.new_full((B,), 1.0)
    lam_hist.append(net.forward_from_ctx(z, t_norm).cpu())

    ts = torch.linspace(0.0, T, N_T + 1)
    X1_hist  = torch.stack([x.cpu() for x in X1_hist])
    X2_hist  = torch.stack([x.cpu() for x in X2_hist])
    lam_hist = torch.stack(lam_hist)
    mix_hist = torch.stack(mix_hist)
    return ts, X1_hist, X2_hist, lam_hist, mix_hist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_SNAP_IDX = None  # filled at runtime


def _snap_indices(N_T: int, n_panels: int = 5) -> list[int]:
    return [int(round(i * N_T / (n_panels - 1))) for i in range(n_panels)]


def _apply_mask(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """X: (n, 3), mask: (n,) bool -> (n_real, 3)."""
    if mask is None:
        return X
    return X[mask.to(X.device)]


def plot_scatter_3d(ts, X1_hist, X2_hist, m1, m2, T, out_path):
    """3D scatter at 6 snapshots (first batch element)."""
    snap = _snap_indices(len(ts) - 1, n_panels=6)
    fig = plt.figure(figsize=(4.5 * len(snap), 4.5))

    for i, k in enumerate(snap):
        ax = fig.add_subplot(1, len(snap), i + 1, projection='3d')
        X1 = _apply_mask(X1_hist[k, 0],
                         m1[0] if m1 is not None else None).numpy()
        X2 = _apply_mask(X2_hist[k, 0],
                         m2[0] if m2 is not None else None).numpy()
        ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2],
                   c='crimson', s=2.0, alpha=0.6, rasterized=True, depthshade=False)
        ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2],
                   c='royalblue', s=2.0, alpha=0.6, rasterized=True, depthshade=False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_zticks([0, 0.5, 1])
        ax.set_title(f"t = {ts[k]:.2f}")
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
        ax.view_init(elev=22.0, azim=-55.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_video(ts, X1_hist, X2_hist, m1, m2, out_path, fps=12):
    """Animate the 3D particle evolution across all rollout steps, save mp4."""
    import matplotlib.animation as animation

    N = len(ts)
    mask1 = m1[0] if m1 is not None else None
    mask2 = m2[0] if m2 is not None else None
    X1_all = [_apply_mask(X1_hist[k, 0], mask1).numpy() for k in range(N)]
    X2_all = [_apply_mask(X2_hist[k, 0], mask2).numpy() for k in range(N)]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_zticks([0, 0.5, 1])
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
    ax.view_init(elev=22.0, azim=-55.0)

    scat1 = ax.scatter(X1_all[0][:, 0], X1_all[0][:, 1], X1_all[0][:, 2],
                       c='crimson', s=2.0, alpha=0.35, depthshade=True)
    scat2 = ax.scatter(X2_all[0][:, 0], X2_all[0][:, 1], X2_all[0][:, 2],
                       c='royalblue', s=2.0, alpha=0.35, depthshade=True)
    title = ax.set_title(f"t = {float(ts[0]):.2f}")

    def update(k):
        scat1._offsets3d = (X1_all[k][:, 0], X1_all[k][:, 1], X1_all[k][:, 2])
        scat2._offsets3d = (X2_all[k][:, 0], X2_all[k][:, 1], X2_all[k][:, 2])
        title.set_text(f"t = {float(ts[k]):.2f}")
        return scat1, scat2, title

    ani = animation.FuncAnimation(fig, update, frames=N, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400,
                                    codec='libx264',
                                    extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(out_path, writer=writer, dpi=150)
    plt.close(fig)


def plot_slices(ts, X1_hist, X2_hist, m1, m2, T, out_path, title="",
                slab_half=0.05):
    """3 rows (one per slab axis) x 5 cols (snapshots)."""
    snap = _snap_indices(len(ts) - 1)
    centres = [0.5, 0.5, 0.5]
    axis_names = ['x1', 'x2', 'x3']
    remaining = [(1, 2), (0, 2), (0, 1)]   # axes to plot per slab axis

    fig, axes = plt.subplots(3, len(snap),
                             figsize=(3.0 * len(snap), 9.0),
                             squeeze=False)

    for row, (slab_ax, (ax_a, ax_b)) in enumerate(zip([0, 1, 2], remaining)):
        lo = centres[slab_ax] - slab_half
        hi = centres[slab_ax] + slab_half
        for col, k in enumerate(snap):
            ax = axes[row][col]
            X1 = _apply_mask(X1_hist[k, 0],
                             m1[0] if m1 is not None else None).numpy()
            X2 = _apply_mask(X2_hist[k, 0],
                             m2[0] if m2 is not None else None).numpy()
            in1 = (X1[:, slab_ax] >= lo) & (X1[:, slab_ax] <= hi)
            in2 = (X2[:, slab_ax] >= lo) & (X2[:, slab_ax] <= hi)
            ax.scatter(X1[in1, ax_a], X1[in1, ax_b],
                       c='crimson', s=4.0, alpha=0.7, rasterized=True)
            ax.scatter(X2[in2, ax_a], X2[in2, ax_b],
                       c='royalblue', s=4.0, alpha=0.7, rasterized=True)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            if col == 0:
                ax.set_ylabel(f"slab: {axis_names[slab_ax]} in "
                              f"[{lo:.2f},{hi:.2f}]\n{axis_names[ax_b]}")
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel(axis_names[ax_a])
            else:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(f"t = {ts[k]:.2f}")

    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_lambda(ts, lam_hist, basis, out_path):
    """lambda_l(t) curves coloured by curl axis (1=e_x, 2=e_y, 3=e_z)."""
    lam = lam_hist[:, 0].numpy()                # (N_T+1, L)
    L = lam.shape[1]
    axis = basis.axis.cpu().numpy()
    ts_np = ts.numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    palette = {1: 'tab:red', 2: 'tab:green', 3: 'tab:blue'}
    for l in range(L):
        ax.plot(ts_np, lam[:, l],
                color=palette[int(axis[l])], alpha=0.5, lw=1.2)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='tab:red',   lw=2, label=r'$\nabla\times(\psi\,\hat e_x)$'),
        Line2D([0], [0], color='tab:green', lw=2, label=r'$\nabla\times(\psi\,\hat e_y)$'),
        Line2D([0], [0], color='tab:blue',  lw=2, label=r'$\nabla\times(\psi\,\hat e_z)$'),
    ]
    ax.legend(handles=handles, loc='upper right')
    ax.axhline(0.0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\lambda_l(t)$')
    ax.set_title(f'Velocity coefficients (L = {L})')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_mix_norm(ts, mix_hist, out_path):
    """Dye mix-norm decay on log-y, square panel -- matches 2D style."""
    mix  = mix_hist[:, 0].numpy()                  # (N_T+1,)  dye_cost = ||.||^2
    ts_np = ts.numpy()
    mix_arr = np.sqrt(np.clip(mix, 0.0, None))     # plot the norm itself

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_box_aspect(1)
    ax.plot(ts_np, mix_arr, color='#1a66cc', linewidth=2,
            label=r'$\Vert \theta_h \Vert_{\dot{H}^{-1}}$')

    idx = max(1, int(0.25 * len(ts_np)))
    valid = mix_arr[idx:] > 0
    if valid.sum() >= 2:
        t_fit  = ts_np[idx:][valid]
        log_mx = np.log(mix_arr[idx:][valid])
        slope, intercept = np.polyfit(t_fit, log_mx, 1)
        k_fit = float(np.exp(intercept)); b_fit = float(slope)
        ax.plot(ts_np, k_fit * np.exp(b_fit * ts_np), 'k--',
                linewidth=1.5,
                label=fr'${k_fit:.2f}\times e^{{{b_fit:.2f}\,t}}$')

    ax.set_yscale('log')
    ax.set_xlabel('t', fontsize=11)
    ax.set_ylabel(r'$\Vert \theta_h \Vert_{\dot{H}^{-1}}$', fontsize=10)
    ax.tick_params(which='both', labelsize=8)
    ax.margins(x=0)
    ax.legend(fontsize=8, loc='best')

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--out_dir',    type=str, default=None,
                   help='Figures dir (default: <ckpt_dir>/figs)')
    p.add_argument('--family',     type=str, default=None,
                   help='Override family name (default: read from ckpt args)')
    p.add_argument('--level_set_K',type=int, default=None)
    p.add_argument('--n_ctx',      type=int, default=1024)
    p.add_argument('--n_traj',     type=int, default=4096,
                   help='Many trajectory particles for nicer figures')
    p.add_argument('--T',          type=float, default=None)
    p.add_argument('--N_T_draw',   type=int,   default=60,
                   help='Sub-steps for the visualisation rollout')
    p.add_argument('--mode',       type=str, default='test',
                   choices=['train', 'test'])
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--device',     type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # -- Load checkpoint ---------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    tr_args = ckpt['args']
    family_name = args.family or tr_args['family']
    level_set_K = args.level_set_K or tr_args.get('level_set_K', 3)
    T = args.T if args.T is not None else tr_args['T']

    family = build_family_3d(family_name, level_set_K=level_set_K)

    basis = DivFreeBasis3D(tr_args['M_basis']).to(args.device)
    dye_cost = DyeMixCost3D(tr_args['K_mix']).to(args.device)
    net = MixingOperatorNet(
        d=3, L=basis.L,
        h=tr_args['h'], n_heads=tr_args['n_heads'],
        n_layers_self=tr_args['n_layers_self'],
        n_layers_joint=tr_args['n_layers_joint'],
        t_emb_dim=tr_args['t_emb_dim'],
        dropout=tr_args.get('dropout', 0.0),
    ).to(args.device)
    net.load_state_dict(ckpt['model'])
    net.eval()

    # -- Draw one test configuration --------------------------------------
    sampler = make_partition_sampler_3d(family, mode=args.mode)
    X1c, X2c, m1c, m2c, X1t, X2t, m1t, m2t = sampler(
        1, args.n_ctx, args.n_traj, device=args.device)

    ts, X1h, X2h, lam_h, mix_h = rollout(
        net, basis, dye_cost,
        X1t, X2t, m1t, m2t, T=T, N_T=args.N_T_draw)

    # -- Output paths -----------------------------------------------------
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint),
                                            "figs")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    stem = f"{family_name}_{args.mode}_{stem}"

    title = f"3D {family_name}  |  mode={args.mode}  |  L={basis.L}  |  N_T={args.N_T_draw}"
    plot_scatter_3d(ts, X1h, X2h, m1t, m2t, T,
                    os.path.join(out_dir, f"{stem}_scatter3d.png"))
    plot_slices(ts, X1h, X2h, m1t, m2t, T,
                os.path.join(out_dir, f"{stem}_slices.png"),
                title=title)
    plot_lambda(ts, lam_h, basis,
                os.path.join(out_dir, f"{stem}_lambda.png"))
    plot_mix_norm(ts, mix_h,
                  os.path.join(out_dir, f"{stem}_mix_norm.png"))
    plot_video(ts, X1h, X2h, m1t, m2t,
               os.path.join(out_dir, f"{stem}_video.mp4"))

    print(f"saved 4 figures + 1 video to {out_dir}/{stem}_*")
    print(f"  mix-norm: start={float(mix_h[0,0]):.4e}   "
          f"end={float(mix_h[-1,0]):.4e}   "
          f"ratio={float(mix_h[-1,0])/max(float(mix_h[0,0]),1e-12):.4e}")


if __name__ == '__main__':
    main()
