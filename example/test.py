"""
test.py  --  Evaluate and visualise a trained two-density mixing operator.

Loads a checkpoint, runs the learned operator on ONE partition example,
and produces a single multi-panel figure:

  Row 0:  Particle evolution  (red/blue scatter at 6 time snapshots)
  Row 1:  Lambda(t) coefficients  |  Mix-norm decay
  Row 2:  Velocity field quiver plots  (optional, --no_quiver to skip)

Usage
-----
  python -m example.test                                     # uses out/ckpt_final.pt
  python -m example.test --checkpoint out/ckpt_050000.pt
  python -m example.test --partition top_bottom --no_quiver
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.basis import DivFreeBasis, DyeMixCost
from model   import MixingOperatorNet
from train   import _lam_at
from model.dataset import get_named_partition, build_family
from model.util import make_lattice


# ---------------------------------------------------------------
# Load model from checkpoint
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


# ---------------------------------------------------------------
# RK4 integration with snapshots
# ---------------------------------------------------------------

@torch.no_grad()
def integrate_with_snapshots(net, basis, z, X1, X2, T, N_T, N_snap=20):
    dt = T / N_T
    snap_interval = max(1, N_T // N_snap)
    snapshots = [(X1.clone(), X2.clone())]
    snap_times = [0.0]

    for s in range(N_T):
        t_s = s * dt
        lam1 = _lam_at(net, z, t_s, T)
        k1_1, k1_2 = basis.velocity(X1, lam1), basis.velocity(X2, lam1)

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


@torch.no_grad()
def compute_velocity_field(net, basis, z, t_val, T, N_quiver=15, device='cpu'):
    g = torch.linspace(0.05, 0.95, N_quiver, device=device)
    x1, x2 = torch.meshgrid(g, g, indexing='ij')
    X_grid = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1).unsqueeze(0)
    lam = _lam_at(net, z, t_val, T)
    v = basis.velocity(X_grid, lam)
    v_np = v[0].cpu().numpy()
    return (
        x1.cpu().numpy(), x2.cpu().numpy(),
        v_np[:, 0].reshape(N_quiver, N_quiver),
        v_np[:, 1].reshape(N_quiver, N_quiver),
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Evaluate a trained two-density mixing operator")
    p.add_argument('--checkpoint', type=str, default='out/ckpt_final.pt')
    p.add_argument('--out_dir',    type=str, default='out')
    p.add_argument('--N_grid',     type=int, default=50)
    p.add_argument('--N_snap',     type=int, default=20)
    p.add_argument('--K_mix',      type=int, default=24)
    p.add_argument('--N_quiver',   type=int, default=15)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--N_T',        type=int, default=None)
    p.add_argument('--no_quiver',  action='store_true')
    p.add_argument('--partition',  type=str, default=None,
                   help='[legacy] left_right, top_bottom, diagonal, radial')
    p.add_argument('--family',     type=str, default=None,
                   choices=['rotation_line', 'general_line',
                            'circle', 'level_set'],
                   help='Partition family (overrides --partition)')
    p.add_argument('--param',      type=str, default=None,
                   help='Comma-separated parameter values for --family '
                        '(e.g. "0.785" or "0.5,0.5,0.25")')
    p.add_argument('--level_set_K', type=int, default=3)
    p.add_argument('--device',     type=str,
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
    print(f"M_basis={M_basis}  L={L}  T={T}  N_T={N_T}")
    print(f"Lattice: {args.N_grid}x{args.N_grid} = {args.N_grid**2} particles")

    dye_cost = DyeMixCost(args.K_mix).to(device)

    # --- Resolve partition source: family+param > legacy partition > ckpt family ---
    family_name = args.family or ckpt_args.get('family', None)
    if family_name and not args.partition:
        family = build_family(family_name, level_set_K=args.level_set_K)
        if args.param:
            vals = [float(v) for v in args.param.split(',')]
        else:
            vals = family.sample_param(mode='train', device=device).tolist()
        if len(vals) != family.dim:
            raise ValueError(
                f"--param has {len(vals)} values but family '{family_name}' "
                f"expects {family.dim}")
        p_tensor = torch.tensor(vals, device=device)
        label    = family.describe(p_tensor)
        part_fn  = lambda X: family.predicate(X, p_tensor)
    else:
        pname = args.partition or 'left_right'
        part_fn = get_named_partition(pname)
        label   = pname

    # --- Prepare lattice and partition ---
    X_lattice = make_lattice(args.N_grid, device=device)  # (1, N^2, 2)
    pts = X_lattice[0]   # (N^2, 2)
    mask = part_fn(pts)

    X1 = pts[mask].unsqueeze(0)    # (1, n1, 2) — blue
    X2 = pts[~mask].unsqueeze(0)   # (1, n2, 2) — red
    part_label = label
    file_tag = (part_label.replace('/', '_').replace(' ', '')
                          .replace('(', '').replace(')', '')
                          .replace(',', '_'))
    print(f"Partition '{part_label}': n1={X1.shape[1]}, n2={X2.shape[1]}")

    # Encode context
    z = net.encode_context(X1, X2)

    # --- Integrate ---
    print("Integrating ODE ...")
    snapshots, snap_times = integrate_with_snapshots(
        net, basis, z, X1, X2, T, N_T, N_snap=args.N_snap)

    # Dye-norm at each snapshot
    dye_norms = [dye_cost(s1, s2).item() for s1, s2 in snapshots]

    # Dense lambda(t)
    print("Evaluating lambda(t) ...")
    dense_times, dense_lambdas = evaluate_lambdas_dense(net, z, T)

    # --- Build figure ---
    n_panels = 6
    n_snaps = len(snapshots)
    if n_snaps <= n_panels:
        sel = list(range(n_snaps))
    else:
        sel = [int(round(i * (n_snaps - 1) / (n_panels - 1)))
               for i in range(n_panels)]
        sel[0] = 0
        sel[-1] = n_snaps - 1
    n_panels = len(sel)

    pt_size = max(1.0, 600 / args.N_grid)
    show_quiver = not args.no_quiver
    n_rows = 3 if show_quiver else 2
    height_ratios = [3, 2, 3] if show_quiver else [3, 2]

    fig = plt.figure(figsize=(3.2 * n_panels, 4.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_panels, height_ratios=height_ratios,
                           hspace=0.35, wspace=0.15)

    # Row 0: particle scatter
    for col, si in enumerate(sel):
        ax = fig.add_subplot(gs[0, col])
        X1s = snapshots[si][0][0].cpu().numpy()
        X2s = snapshots[si][1][0].cpu().numpy()
        ax.scatter(X1s[:, 0], X1s[:, 1], c='#1f77b4', s=pt_size,
                   alpha=0.8, edgecolors='none', rasterized=True)
        ax.scatter(X2s[:, 0], X2s[:, 1], c='#d62728', s=pt_size,
                   alpha=0.8, edgecolors='none', rasterized=True)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_title(f't = {snap_times[si]:.2f}', fontsize=9)
        ax.tick_params(labelsize=6)
        if col > 0:
            ax.set_yticklabels([])

    # Row 1 left: lambda(t)
    mid = n_panels // 2
    ax_lam = fig.add_subplot(gs[1, :mid])
    ms_np = basis.ms.cpu().numpy().astype(int)
    ns_np = basis.ns.cpu().numpy().astype(int)
    cmap = plt.cm.tab10
    for l_idx in range(L):
        c = cmap(l_idx % 10)
        label = f'$\\lambda_{{{ms_np[l_idx]},{ns_np[l_idx]}}}$' if L <= 12 else None
        ax_lam.plot(dense_times, dense_lambdas[:, l_idx],
                    color=c, linewidth=1.0, alpha=0.8, label=label)
    ax_lam.set_xlabel('t', fontsize=10)
    ax_lam.set_ylabel(r'$\lambda_\ell(t)$', fontsize=10)
    ax_lam.set_title(r'Basis coefficients $\lambda(t)$', fontsize=10, fontweight='bold')
    ax_lam.grid(True, alpha=0.3)
    if L <= 12:
        ax_lam.legend(fontsize=7, ncol=2, loc='best')

    # Row 1 right: mix-norm decay
    ax_mix = fig.add_subplot(gs[1, mid:])
    ax_mix.plot(snap_times, dye_norms, 'o-', color='#2ca02c', linewidth=2, markersize=4)
    ax_mix.set_xlabel('t', fontsize=10)
    ax_mix.set_ylabel(r'$\|f\|^2_{\dot{H}^{-1}}$', fontsize=10)
    ax_mix.set_title('Mix-norm decay', fontsize=10, fontweight='bold')
    ax_mix.grid(True, alpha=0.3)
    ax_mix.annotate(f'{dye_norms[0]:.4f}', xy=(snap_times[0], dye_norms[0]),
                    xytext=(12, -8), textcoords='offset points', fontsize=8, color='#d62728')
    ax_mix.annotate(f'{dye_norms[-1]:.4f}', xy=(snap_times[-1], dye_norms[-1]),
                    xytext=(-50, 8), textcoords='offset points', fontsize=8, color='#1f77b4')

    # Row 2: quiver plots
    if show_quiver:
        all_speeds = []
        quiver_data = []
        for si in sel:
            gx, gy, vx, vy = compute_velocity_field(
                net, basis, z, snap_times[si], T, args.N_quiver, device)
            speed = np.sqrt(vx**2 + vy**2)
            all_speeds.append(speed.max())
            quiver_data.append((gx, gy, vx, vy, speed))
        max_speed = max(all_speeds) if all_speeds else 1.0

        for col, (si, qd) in enumerate(zip(sel, quiver_data)):
            gx, gy, vx, vy, speed = qd
            ax = fig.add_subplot(gs[2, col])
            ax.quiver(gx, gy, vx, vy, speed, cmap='coolwarm',
                      clim=(0, max_speed), scale=max_speed * 15,
                      scale_units='width', width=0.015, headwidth=3, alpha=0.85)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_title(f't = {snap_times[si]:.2f}', fontsize=9)
            ax.tick_params(labelsize=6)
            if col > 0:
                ax.set_yticklabels([])

    save_path = os.path.join(args.out_dir,
                             f"test_result_M{M_basis}_{file_tag}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {save_path}")
    plt.close()

    # Summary
    ratio = dye_norms[-1] / dye_norms[0] if dye_norms[0] > 0 else 0.0
    print(f"Mix-norm: {dye_norms[0]:.6f} -> {dye_norms[-1]:.6f} "
          f"({(1 - ratio) * 100:.1f}% reduction)")


if __name__ == '__main__':
    main()
