"""
test3d_density.py -- Eulerian density visualisation of the 3D network.

Loads a trained 3D checkpoint, queries lambda(t) at the N_T midpoints from
the network (the optimal control), then evolves the continuous density
theta(t, x, y, z) on a cell-centred cube grid using the same PDE solver as
the 2D case: MUSCL (minmod) upwind + Heun (RK2), CFL-substepping, on a
MAC staggered grid with face velocities assembled from basis3d's
div-free basis.

Outputs (next to the checkpoint):
  <stem>_density3d.png        6 3D snapshots of theta
  <stem>_density3d_mix.png    Eulerian mix-norm decay (2D-style, square)
  <stem>_density3d.mp4        video of theta evolution (all N_T frames)

Usage
-----
  python -m example.test3d_density --checkpoint out/3d_rotation_plane_d2/ckpt_final.pt
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


# ---------------------------------------------------------------------------
# Network -> lambda(t) schedule at N_T midpoints
# ---------------------------------------------------------------------------

@torch.no_grad()
def network_lambda_schedule(net, X1c, X2c, m1c, m2c,
                            T: float, N_T: int) -> torch.Tensor:
    """lambda^NN(t_k) at the N_T sub-interval midpoints, shape (N_T, L)."""
    z = net.encode_context(X1c, X2c, m1c, m2c)
    dt = T / N_T
    lams = []
    for s in range(N_T):
        t_val = s * dt + 0.5 * dt
        t_norm = z.new_full((z.shape[0],), t_val / T)
        lam_s = net.forward_from_ctx(z, t_norm)[0].cpu()        # (L,)
        lams.append(lam_s)
    return torch.stack(lams, dim=0)


# ---------------------------------------------------------------------------
# Simulate and record theta at every control step
# ---------------------------------------------------------------------------

def simulate_with_trajectory(lam: torch.Tensor,
                             rho0: torch.Tensor,
                             E1: torch.Tensor,
                             E2: torch.Tensor,
                             E3: torch.Tensor,
                             T: float, cfl: float = 0.5):
    """Return (rho_traj (N_T+1, N, N, N), times (N_T+1,))."""
    N_T = lam.shape[0]
    Nx  = rho0.shape[0]
    h   = 1.0 / Nx
    dt  = T / N_T
    rho = rho0.clone()
    traj  = [rho.clone()]
    times = [0.0]
    for k in range(N_T):
        v1, v2, v3 = _assemble_velocity_3d(lam[k], E1, E2, E3)
        rho = advect_interval_3d(rho, v1, v2, v3, dt, h, cfl=cfl)
        traj.append(rho.clone())
        times.append((k + 1) * dt)
    return torch.stack(traj, dim=0), np.asarray(times)


# ---------------------------------------------------------------------------
# 3D density rendering (overlapping-scatter, tuned to read as a continuous
# two-phase volume rather than a lattice of discrete dots)
# ---------------------------------------------------------------------------

_RED  = np.array([0.84, 0.15, 0.16])
_BLUE = np.array([0.12, 0.47, 0.71])


def _setup_3d_axes(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
    ax.view_init(elev=22.0, azim=-55.0)


def _theta_points(theta: np.ndarray, stride: int,
                  a_min: float = 0.18, a_max: float = 0.73):
    """
    Alpha-fade scatter: cells far from theta=0.5 are opaque (pure red/blue);
    cells near the interface fade to translucent, which lets the lattice
    pattern show through and separates the two phases visually.
    """
    N = theta.shape[0]
    idx = np.arange(0, N, stride)
    gg  = (idx + 0.5) / N
    X, Y, Z = np.meshgrid(gg, gg, gg, indexing='ij')
    th = np.clip(theta[::stride, ::stride, ::stride], 0.0, 1.0).reshape(-1)
    rgb = th[:, None] * _RED + (1.0 - th[:, None]) * _BLUE
    a   = a_min + (a_max - a_min) * np.abs(2.0 * th - 1.0)
    rgba = np.concatenate([rgb, a[:, None]], axis=-1)
    xyz  = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=-1)
    return xyz, rgba


def plot_density_snapshots(times, rho_traj, out_path,
                           n_panels: int = 6, stride: int = 3,
                           s: float = 14.0):
    N_T = rho_traj.shape[0] - 1
    sel = [int(round(i * N_T / (n_panels - 1))) for i in range(n_panels)]

    fig = plt.figure(figsize=(4.5 * n_panels, 4.5))
    for i, k in enumerate(sel):
        ax = fig.add_subplot(1, n_panels, i + 1, projection='3d')
        xyz, rgba = _theta_points(rho_traj[k].cpu().numpy(), stride=stride)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c=rgba, s=s, depthshade=True, rasterized=True,
                   edgecolors='none', marker='o')
        _setup_3d_axes(ax)
        ax.set_title(f't = {float(times[k]):.2f}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_density_video(times, rho_traj, out_path,
                       stride: int = 2, s: float = 14.0, fps: int = 12):
    import matplotlib.animation as animation

    N = rho_traj.shape[0]
    frames = [_theta_points(rho_traj[k].cpu().numpy(), stride=stride)
              for k in range(N)]

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection='3d')
    _setup_3d_axes(ax)
    xyz0, rgba0 = frames[0]
    scat = ax.scatter(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2],
                      c=rgba0, s=s, depthshade=True,
                      edgecolors='none', marker='o')
    title = ax.set_title(f't = {float(times[0]):.2f}')

    def update(k):
        xyz, rgba = frames[k]
        scat._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        scat.set_color(rgba)
        title.set_text(f't = {float(times[k]):.2f}')
        return scat, title

    ani = animation.FuncAnimation(fig, update, frames=N, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400, codec='libx264',
                                    extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(out_path, writer=writer, dpi=150)
    plt.close(fig)


def plot_lambda(times_lam, lam, basis, out_path):
    """lambda_l(t) curves coloured by curl axis (1=e_x, 2=e_y, 3=e_z)."""
    from matplotlib.lines import Line2D
    lam_np = lam.cpu().numpy() if torch.is_tensor(lam) else np.asarray(lam)
    L = lam_np.shape[1]
    axis = basis.axis.cpu().numpy()
    ts_np = np.asarray(times_lam)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    palette = {1: 'tab:red', 2: 'tab:green', 3: 'tab:blue'}
    for l in range(L):
        ax.plot(ts_np, lam_np[:, l],
                color=palette[int(axis[l])], alpha=0.5, lw=1.2)
    handles = [
        Line2D([0], [0], color='tab:red',   lw=2,
               label=r'$\nabla\times(\psi\,\hat e_x)$'),
        Line2D([0], [0], color='tab:green', lw=2,
               label=r'$\nabla\times(\psi\,\hat e_y)$'),
        Line2D([0], [0], color='tab:blue',  lw=2,
               label=r'$\nabla\times(\psi\,\hat e_z)$'),
    ]
    ax.legend(handles=handles, loc='upper right')
    ax.axhline(0.0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\lambda_l(t)$')
    ax.set_title(f'Velocity coefficients (L = {L})')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_mix_norm(times, mix_curve, out_path):
    """2D-style square panel, log-y, exp fit."""
    t = np.asarray(times)
    m = np.sqrt(np.clip(np.asarray(mix_curve), 0.0, None))

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_box_aspect(1)
    ax.plot(t, m, color='#1a66cc', linewidth=2,
            label=r'$\Vert \theta_h \Vert_{\dot{H}^{-1}(\Omega)}$')

    idx = max(1, int(0.25 * len(t)))
    valid = m[idx:] > 0
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(t[idx:][valid],
                                      np.log(m[idx:][valid]), 1)
        k_fit = float(np.exp(intercept)); b_fit = float(slope)
        ax.plot(t, k_fit * np.exp(b_fit * t), 'k--', linewidth=1.5,
                label=fr'${k_fit:.2f}\times e^{{{b_fit:.2f}\,t}}$')

    ax.set_yscale('log')
    ax.set_xlabel('t', fontsize=11)
    ax.set_ylabel(r'$\Vert \theta_h \Vert_{\dot{H}^{-1}(\Omega)}$',
                  fontsize=10)
    ax.tick_params(which='both', labelsize=8)
    ax.margins(x=0)
    ax.legend(fontsize=8, loc='best')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sample_partition(family, mode: str, n_ctx: int, device):
    """Sample one family parameter p and split a fresh cloud of context pts."""
    p = family.sample_param(mode=mode, device=device)
    X = torch.rand(n_ctx, 3, device=device)
    mask = family.predicate(X, p)
    # Pad to same length: we keep X1_ctx and X2_ctx as lists of real pts.
    X1 = X[mask].unsqueeze(0)
    X2 = X[~mask].unsqueeze(0)
    m1 = torch.ones(X1.shape[1], dtype=torch.bool, device=device).unsqueeze(0)
    m2 = torch.ones(X2.shape[1], dtype=torch.bool, device=device).unsqueeze(0)
    return p, X1, X2, m1, m2


def main():
    p_ = argparse.ArgumentParser()
    p_.add_argument('--checkpoint', type=str, required=True)
    p_.add_argument('--out_dir',    type=str, default=None)
    p_.add_argument('--family',     type=str, default=None)
    p_.add_argument('--level_set_K',type=int, default=None)
    p_.add_argument('--n_ctx',      type=int, default=1024)
    p_.add_argument('--T',          type=float, default=None)
    p_.add_argument('--N_T_sim',    type=int,   default=60,
                    help='PDE control step count (frames in the video)')
    p_.add_argument('--N_x',        type=int,   default=48,
                    help='PDE grid resolution for the snapshot PNG')
    p_.add_argument('--N_x_vid',    type=int,   default=32,
                    help='PDE grid resolution for the video (lower for speed)')
    p_.add_argument('--s',          type=float, default=14.0,
                    help='Scatter marker size (shared by PNG and MP4)')
    p_.add_argument('--stride',     type=int,   default=2,
                    help='Subsampling stride on the N_x_vid grid '
                         '(shared by PNG and MP4 so their colours match)')
    p_.add_argument('--K_mix',      type=int,   default=None)
    p_.add_argument('--mode',       type=str,   default='test',
                    choices=['train', 'test'])
    p_.add_argument('--seed',       type=int,   default=0)
    p_.add_argument('--cfl',        type=float, default=0.5)
    p_.add_argument('--device',     type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p_.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    # ---- load ckpt + network --------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tr_args = ckpt['args']
    family_name = args.family or tr_args['family']
    level_set_K = args.level_set_K or tr_args.get('level_set_K', 3)
    T       = args.T      if args.T      is not None else tr_args['T']
    K_mix   = args.K_mix  if args.K_mix  is not None else tr_args.get('K_mix', 8)
    M_basis = tr_args['M_basis']

    family = build_family_3d(family_name, level_set_K=level_set_K)

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

    # ---- sample partition & rho0 grid ----------------------------------
    p_vec, X1c, X2c, m1c, m2c = sample_partition(
        family, args.mode, args.n_ctx, device)
    print(f'Family: {family_name}  param = {family.describe(p_vec)}')

    rho0 = family_rho0_grid_3d(family, p_vec, args.N_x, device=device)

    # ---- network lambda schedule ---------------------------------------
    print(f'Querying network for lambda^NN at N_T={args.N_T_sim} midpoints...')
    lam = network_lambda_schedule(net, X1c, X2c, m1c, m2c,
                                  T=T, N_T=args.N_T_sim)
    lam_d = lam.to(device=device, dtype=torch.float64)

    # ---- 3D Eulerian transport -----------------------------------------
    print(f'Building face-grid basis (M={M_basis}, Nx={args.N_x})...')
    E1, E2, E3 = _basis_on_faces_3d(M_basis, args.N_x,
                                    device=device, dtype=torch.float64)
    print(f'Running 3D MUSCL+Heun transport (T={T}, cfl={args.cfl})...')
    rho_traj, times = simulate_with_trajectory(
        lam_d, rho0, E1, E2, E3, T=T, cfl=args.cfl)

    # ---- grid mix-norm curve (dye, matches train3d loss) --------------
    mix_curve = np.asarray([
        float(grid_mix_norm_sq_3d(r, K_max=K_mix, dye=True))
        for r in rho_traj])
    mass0 = float(rho_traj[0].sum()) * (1.0 / args.N_x) ** 3
    massT = float(rho_traj[-1].sum()) * (1.0 / args.N_x) ** 3
    print(f'  mix^2 start = {mix_curve[0]:.4e}   end = {mix_curve[-1]:.4e}   '
          f'ratio = {mix_curve[-1]/max(mix_curve[0],1e-16):.4e}')
    print(f'  mass        = {mass0:.6f} -> {massT:.6f}   '
          f'drift {massT-mass0:+.2e}')
    print(f'  theta range at T: [{float(rho_traj[-1].min()):.3f}, '
          f'{float(rho_traj[-1].max()):.3f}]')

    # ---- figures --------------------------------------------------------
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint),
                                            'figs')
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    stem = f'{family_name}_{args.mode}_{stem}_density3d'

    # Downsample theta to N_x_vid so PNG and MP4 render from the SAME grid
    # at the SAME stride -- their two-phase colours then read identically.
    if args.N_x_vid < args.N_x and args.N_x % args.N_x_vid == 0:
        blk = args.N_x // args.N_x_vid
        vid_traj = rho_traj.reshape(rho_traj.shape[0],
                                    args.N_x_vid, blk, args.N_x_vid, blk,
                                    args.N_x_vid, blk).mean(dim=(2, 4, 6))
    else:
        vid_traj = rho_traj

    plot_density_snapshots(times, vid_traj,
                           os.path.join(out_dir, f'{stem}.png'),
                           n_panels=6, stride=args.stride, s=args.s)
    plot_density_video(times, vid_traj,
                       os.path.join(out_dir, f'{stem}.mp4'),
                       stride=args.stride, s=args.s)
    plot_mix_norm(times, mix_curve,
                  os.path.join(out_dir, f'{stem}_mix.png'))

    # lambda(t) -- use the N_T midpoint times that match the schedule.
    dt = T / args.N_T_sim
    t_lam = (np.arange(args.N_T_sim) + 0.5) * dt
    plot_lambda(t_lam, lam, basis,
                os.path.join(out_dir, f'{stem}_lambda.png'))

    print(f'Saved figures + video + lambda to {out_dir}/{stem}*')


if __name__ == '__main__':
    main()
