"""
make_videos_2d.py -- Per-model particle + Eulerian MP4s for 2D checkpoints.

For a single trained 2D mixing-operator checkpoint this produces two videos
from one test-mode parameter sample:

  1. test_multi_<family>.mp4
       red/blue particle lattice advected under the network velocity.
       Parallels the snapshots shown in test_multi_<family>.png.

  2. compare_<family>_multi.mp4
       Eulerian two-phase density (Nx x Nx) evolved by the SAME Eulerian
       transport solver that test_multi_<family>.png's row-"Network" uses,
       driven by lambda^NN(t). Parallels the second row of
       compare_<family>_multi.png.

Usage
-----
    python -m example.make_videos_2d \
        --checkpoint ../results/2d/level_set_d6/ckpt_final.pt \
        --out_particle ../results/2d/level_set_d6 \
        --out_eulerian ../results/2d/eulerian \
        --device cpu
"""

from __future__ import annotations

# Allow both module execution and direct execution of this example.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from model.dataset import build_family

from example.test_multi import load_model, integrate_with_snapshots

from model.eulerian.pde_solver import _basis_on_faces
from example.compare_pde_particle import (
    family_lattice,
    family_rho0_grid,
    network_lambda_schedule,
    simulate_eulerian_full,
)


RED_RGB  = np.array([0.84, 0.15, 0.16])
BLUE_RGB = np.array([0.12, 0.47, 0.71])


def render_particle_video(snapshots, snap_times, out_path: str,
                          pt_size: float = 3.0, fps: int = 20):
    X1_0, X2_0 = snapshots[0]
    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    sc1 = ax.scatter(X1_0[0, :, 0].cpu().numpy(),
                     X1_0[0, :, 1].cpu().numpy(),
                     c='#1f77b4', s=pt_size, alpha=0.85,
                     edgecolors='none', rasterized=True)
    sc2 = ax.scatter(X2_0[0, :, 0].cpu().numpy(),
                     X2_0[0, :, 1].cpu().numpy(),
                     c='#d62728', s=pt_size, alpha=0.85,
                     edgecolors='none', rasterized=True)
    title = ax.set_title(f't = {snap_times[0]:.2f}', fontsize=13)

    def update(i):
        X1_i, X2_i = snapshots[i]
        sc1.set_offsets(np.c_[X1_i[0, :, 0].cpu().numpy(),
                              X1_i[0, :, 1].cpu().numpy()])
        sc2.set_offsets(np.c_[X2_i[0, :, 0].cpu().numpy(),
                              X2_i[0, :, 1].cpu().numpy()])
        title.set_text(f't = {snap_times[i]:.2f}')
        return sc1, sc2, title

    anim = FuncAnimation(fig, update, frames=len(snapshots),
                         interval=1000 // fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2500)
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)


def render_eulerian_video(rho_traj, times, out_path: str, fps: int = 20):
    def _to_rgb(rho):
        r = rho.cpu().numpy() if torch.is_tensor(rho) else np.asarray(rho)
        a = np.clip(r[..., None], 0.0, 1.0)
        rgb = a * RED_RGB + (1.0 - a) * BLUE_RGB
        return np.transpose(rgb, (1, 0, 2))

    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(_to_rgb(rho_traj[0]), origin='lower',
                   extent=[0, 1, 0, 1], aspect='equal',
                   interpolation='nearest')
    title = ax.set_title(f't = {float(times[0]):.2f}', fontsize=13)

    def update(i):
        im.set_data(_to_rgb(rho_traj[i]))
        title.set_text(f't = {float(times[i]):.2f}')
        return im, title

    anim = FuncAnimation(fig, update, frames=len(rho_traj),
                         interval=1000 // fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2500)
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--checkpoint',   type=str, required=True)
    p.add_argument('--out_particle', type=str, required=True,
                   help='Directory for test_multi_<family>.mp4')
    p.add_argument('--out_eulerian', type=str, required=True,
                   help='Directory for compare_<family>_multi.mp4')
    p.add_argument('--param_mode',   type=str, default='test',
                   choices=['train', 'test'])
    p.add_argument('--N_grid',       type=int, default=50,
                   help='Particle-lattice resolution (N_grid x N_grid)')
    p.add_argument('--N_x',          type=int, default=128,
                   help='Eulerian grid resolution')
    p.add_argument('--N_T',          type=int, default=200,
                   help='Sub-intervals for velocity and animation frames')
    p.add_argument('--N_snap',       type=int, default=200,
                   help='Particle snapshots kept for animation')
    p.add_argument('--fps',          type=int, default=20)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--device',       type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    net, basis, ckpt_args = load_model(args.checkpoint, device)
    T           = ckpt_args.get('T', 5.0)
    M_basis     = ckpt_args.get('M_basis', 2)
    family_name = ckpt_args.get('family', 'rotation_line')
    level_set_K = ckpt_args.get('level_set_K', 3)
    family = build_family(family_name, level_set_K=level_set_K)
    ho = ckpt_args.get('hole_override', None)
    if ho:
        family.hole = json.loads(ho)

    p_vec = family.sample_param(mode=args.param_mode, device=device)
    print(f'Family     : {family.name}')
    print(f'Parameter  : {family.describe(p_vec)}  (mode={args.param_mode})')
    print(f'M_basis={M_basis}  L={basis.L}  T={T}  N_T={args.N_T}')

    X1, X2 = family_lattice(family, p_vec, args.N_grid, device=device)
    z = net.encode_context(X1, X2)

    os.makedirs(args.out_particle, exist_ok=True)
    os.makedirs(args.out_eulerian, exist_ok=True)

    print('Integrating particle trajectory...')
    snapshots, snap_times = integrate_with_snapshots(
        net, basis, z, X1, X2, T, args.N_T, N_snap=args.N_snap)
    out_p = os.path.join(args.out_particle,
                         f'test_multi_{family.name}.mp4')
    print(f'Rendering particle video ({len(snapshots)} frames) -> {out_p}')
    render_particle_video(snapshots, snap_times, out_p,
                          pt_size=max(1.0, 600 / args.N_grid),
                          fps=args.fps)

    print('Reading lambda^NN schedule...')
    lam_nn = network_lambda_schedule(net, X1, X2, T, args.N_T)
    lam_d  = lam_nn.to(device=device, dtype=torch.float64)

    print(f'Building Eulerian rho0 on N_x={args.N_x} grid...')
    rho0 = family_rho0_grid(family, p_vec, args.N_x, device=device)
    E1, E2 = _basis_on_faces(M_basis, args.N_x,
                             device=device, dtype=torch.float64)

    print('Running Eulerian transport...')
    rho_traj, times = simulate_eulerian_full(lam_d, rho0, E1, E2, T=T)
    out_e = os.path.join(args.out_eulerian,
                         f'compare_{family.name}_multi.mp4')
    print(f'Rendering Eulerian video ({len(rho_traj)} frames) -> {out_e}')
    render_eulerian_video(rho_traj, times, out_e, fps=args.fps)

    print('Done.')


if __name__ == '__main__':
    main()
