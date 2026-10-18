"""
train3d.py  --  Training script for the 3D two-density fluid mixing operator.

3D analogue of train.py.  Domain is Omega = (0,1)^3, velocity is expanded
in the div-free basis from model/basis3d.py built as curl(psi e_k) with
psi = sin(m pi x) sin(n pi y) sin(l pi z) and k in {1,2,3} (L = 3 M^3),
mixing cost uses the 3D H^{-1} Neumann cosine basis.

  min_{theta}  E_{partition} [
      lam_kin * int_0^T (1/N) sum_{i=1}^{N} ||v(t, X_i(t))||^2 dt
    + lam_mix * ||f(T)||^2_{H^{-1}(Omega)}
  ]

  with N = n1 + n2 the pooled trajectory cloud (plain kinetic energy under
  volume preservation) and centered signed weights in the H^{-1} estimator.

Usage
-----
  python train3d.py --family rotation_plane
  python train3d.py --family level_set_3d --level_set_K 3 --n_iter 80000
"""

from __future__ import annotations

import os
import math
import json
import argparse
import time

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched

from model.basis3d import DivFreeBasis3D, DyeMixCost3D
from model     import MixingOperatorNet
from model.dataset3d import build_family_3d, make_partition_sampler_3d
from model.util import setup_logger, log_run_header, log_training_summary


# ===========================================================================
# ODE Integrator (RK4)
# ===========================================================================

def _lam_at(net: MixingOperatorNet, z: torch.Tensor,
            t_val: float, T: float) -> torch.Tensor:
    B = z.shape[0]
    t_norm = z.new_full((B,), t_val / T)
    return net.forward_from_ctx(z, t_norm)


def rk4_integrate(
    net:   MixingOperatorNet,
    basis: DivFreeBasis3D,
    z:     torch.Tensor,        # (B, h)
    X1_0:  torch.Tensor,        # (B, n1, 3)
    X2_0:  torch.Tensor,        # (B, n2, 3)
    mask1: torch.Tensor | None,
    mask2: torch.Tensor | None,
    T:     float = 1.0,
    N_T:   int   = 30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate BOTH particle sets with the SAME velocity field (RK4).
    Positions are clamped to [0,1]^3 after each step to absorb numerical
    drift across the cube boundary (where v . n is already zero).
    """
    dt  = T / N_T
    X1  = X1_0.clone()
    X2  = X2_0.clone()
    B   = X1.shape[0]
    kin1 = X1.new_zeros(B)
    kin2 = X2.new_zeros(B)

    if mask1 is not None:
        n1_real = mask1.float().sum(-1).clamp(min=1)
    else:
        n1_real = torch.tensor(X1.shape[1], device=X1.device, dtype=X1.dtype)
    if mask2 is not None:
        n2_real = mask2.float().sum(-1).clamp(min=1)
    else:
        n2_real = torch.tensor(X2.shape[1], device=X2.device, dtype=X2.dtype)

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

        v1 = (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6.0
        v2 = (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6.0
        X1 = (X1 + dt * v1).clamp(0.0, 1.0)
        X2 = (X2 + dt * v2).clamp(0.0, 1.0)

        ke1 = (k2_1 ** 2).sum(-1)
        ke2 = (k2_2 ** 2).sum(-1)
        if mask1 is not None:
            ke1 = ke1 * mask1.float()
        if mask2 is not None:
            ke2 = ke2 * mask2.float()
        # Pooled normalization: E_kin = (dt/N) sum over ALL trajectory particles,
        # N = n1 + n2 (Monte Carlo estimate of int_0^T int_Omega |v|^2 dx dt).
        N_real = n1_real + n2_real
        kin1 += (ke1.sum(-1) / N_real) * dt
        kin2 += (ke2.sum(-1) / N_real) * dt

    return X1, X2, kin1, kin2


# ===========================================================================
# Loss
# ===========================================================================

def compute_loss(
    net:      MixingOperatorNet,
    basis:    DivFreeBasis3D,
    dye_cost: DyeMixCost3D,
    X1_ctx, X2_ctx, m1c, m2c,
    X1_traj, X2_traj, m1t, m2t,
    T:        float,
    N_T:      int,
    lam_mix:  float,
    lam_kin:  float,
) -> tuple[torch.Tensor, dict]:
    z = net.encode_context(X1_ctx, X2_ctx, m1c, m2c)
    X1_T, X2_T, kin1, kin2 = rk4_integrate(
        net, basis, z, X1_traj, X2_traj, m1t, m2t, T=T, N_T=N_T)
    mix_en = dye_cost(X1_T, X2_T, m1t, m2t)

    loss_mix = lam_mix * mix_en.mean()
    loss_kin = lam_kin * (kin1 + kin2).mean()
    loss     = loss_mix + loss_kin

    info = {
        'loss':     loss.detach().item(),
        'mix_cost': mix_en.mean().detach().item(),
        'kin1':     kin1.mean().detach().item(),
        'kin2':     kin2.mean().detach().item(),
    }
    return loss, info


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate(net, basis, dye_cost, sampler, args, device, n_eval: int = 8):
    net.eval()
    X1c, X2c, m1c, m2c, X1t, X2t, m1t, m2t = sampler(
        n_eval, args.n_ctx, args.n_traj, device=device)
    z = net.encode_context(X1c, X2c, m1c, m2c)
    X1_T, X2_T, kin1, kin2 = rk4_integrate(
        net, basis, z, X1t, X2t, m1t, m2t, T=args.T, N_T=args.N_T)
    mix_en = dye_cost(X1_T, X2_T, m1t, m2t)
    net.train()
    return {
        'mix': mix_en.mean().item(),
        'kin': (kin1 + kin2).mean().item(),
    }


# ===========================================================================
# Training loop
# ===========================================================================

def train(args):
    device = args.device

    hole_override = json.loads(args.hole_override) if args.hole_override else None
    family = build_family_3d(args.family, level_set_K=args.level_set_K,
                             hole_override=hole_override)

    if args.out_dir == 'out':
        args.out_dir = os.path.join('out', f"3d_{family.name}_d{family.dim}")
    os.makedirs(args.out_dir, exist_ok=True)

    logger = setup_logger(args.out_dir, name="train")
    train_start_time = time.time()

    basis    = DivFreeBasis3D(args.M_basis).to(device)
    dye_cost = DyeMixCost3D(args.K_mix).to(device)
    train_sampler = make_partition_sampler_3d(family, mode='train')
    test_sampler  = make_partition_sampler_3d(family, mode='test')

    L   = basis.L
    net = MixingOperatorNet(
        d              = 3,
        L              = L,
        h              = args.h,
        n_heads        = args.n_heads,
        n_layers_self  = args.n_layers_self,
        n_layers_joint = args.n_layers_joint,
        t_emb_dim      = args.t_emb_dim,
        dropout        = args.dropout,
    ).to(device)

    log_run_header(logger, args, net, family=family)
    logger.info(f"  Div-free basis L   : {L}  (M_max={args.M_basis})   [3D curl(psi e_k), psi=sin*sin*sin]")
    logger.info(f"  Spectral modes K   : {dye_cost.K}  (K_max={args.K_mix})   [3D Neumann]")
    logger.info("")

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    warmup_iters = max(1, args.n_iter // 50)
    warmup_sched = sched.LinearLR(optimizer, start_factor=1e-2,
                                   end_factor=1.0,
                                   total_iters=warmup_iters)
    cosine_sched = sched.CosineAnnealingLR(optimizer, T_max=args.n_iter,
                                            eta_min=args.lr * 0.01)
    scheduler = sched.SequentialLR(optimizer,
                                    schedulers=[warmup_sched, cosine_sched],
                                    milestones=[warmup_iters])

    start_iter = 0
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt.get('iteration', 0)
        logger.info(f"Resumed from {args.checkpoint}  (iter {start_iter})")

    net.train()
    log_loss = log_mix = log_kin1 = log_kin2 = 0.0
    t_start = time.time()

    for it in range(start_iter, args.n_iter):
        X1c, X2c, m1c, m2c, X1t, X2t, m1t, m2t = train_sampler(
            args.B_dist, args.n_ctx, args.n_traj, device=device)

        optimizer.zero_grad()
        loss, info = compute_loss(
            net, basis, dye_cost,
            X1c, X2c, m1c, m2c,
            X1t, X2t, m1t, m2t,
            T=args.T, N_T=args.N_T,
            lam_mix=args.lam_mix, lam_kin=args.lam_kin,
        )

        if torch.isfinite(loss):
            loss.backward()
            effective_clip = args.grad_clip / math.sqrt(L)
            total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), effective_clip)
            # Skip the update on non-finite gradients as well: a NaN/inf grad
            # passed through clip_grad_norm_ would poison the parameters
            # permanently even when the loss itself is still finite.
            if torch.isfinite(total_norm):
                optimizer.step()
            else:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        scheduler.step()

        log_loss += info['loss']
        log_mix  += info['mix_cost']
        log_kin1 += info['kin1']
        log_kin2 += info['kin2']

        if (it + 1) % args.log_every == 0:
            n = args.log_every
            elapsed = time.time() - t_start
            logger.info(
                f"iter {it+1:6d}/{args.n_iter} | "
                f"loss {log_loss/n:.4f} | "
                f"mix {log_mix/n:.4f} | "
                f"kin1 {log_kin1/n:.4f} kin2 {log_kin2/n:.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.0f}s"
            )
            log_loss = log_mix = log_kin1 = log_kin2 = 0.0
            t_start = time.time()

        eval_interval = args.eval_test_every or args.eval_every
        if (it + 1) % eval_interval == 0:
            tr = evaluate(net, basis, dye_cost, train_sampler, args, device)
            te = evaluate(net, basis, dye_cost, test_sampler,  args, device)
            logger.info(
                f"  [eval train] mix={tr['mix']:.4f}  kin={tr['kin']:.4f}  |  "
                f"[eval test ] mix={te['mix']:.4f}  kin={te['kin']:.4f}"
            )

        if (it + 1) % args.save_every == 0:
            path = os.path.join(args.out_dir, f"ckpt_{it+1:06d}.pt")
            torch.save({
                'model':     net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': it + 1,
                'args':      vars(args),
            }, path)
            logger.info(f"  [saved] {path}")

    path = os.path.join(args.out_dir, "ckpt_final.pt")
    torch.save({'model': net.state_dict(), 'args': vars(args)}, path)

    log_training_summary(logger, train_start_time,
                         args.n_iter - start_iter, path)


# ===========================================================================
# CLI
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Train 3D two-density fluid mixing operator")

    # Model
    p.add_argument('--M_basis',       type=int,   default=2,
                   help='Div-free basis wavenumber; L = 3*M_basis^3')
    p.add_argument('--K_mix',         type=int,   default=8,
                   help='Spectral truncation for dye mix cost (3D)')
    p.add_argument('--h',             type=int,   default=256)
    p.add_argument('--n_heads',       type=int,   default=4)
    p.add_argument('--n_layers_self', type=int,   default=2)
    p.add_argument('--n_layers_joint',type=int,   default=2)
    p.add_argument('--t_emb_dim',     type=int,   default=64)
    p.add_argument('--dropout',       type=float, default=0.0)

    # ODE
    p.add_argument('--T',   type=float, default=5.0)
    p.add_argument('--N_T', type=int,   default=30)

    # Data  (4x more particles than 2D default, 3D volume scaling)
    p.add_argument('--n_ctx',  type=int, default=1024)
    p.add_argument('--n_traj', type=int, default=1024)
    p.add_argument('--B_dist', type=int, default=4)

    # Loss weights
    p.add_argument('--lam_mix', type=float, default=1.0)
    p.add_argument('--lam_kin', type=float, default=1e-3)

    # Optimisation
    p.add_argument('--n_iter',    type=int,   default=80000)
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)

    # Partition family
    p.add_argument('--family', type=str, required=True,
                   choices=['rotation_plane', 'general_plane',
                            'sphere', 'level_set_3d'])
    p.add_argument('--level_set_K', type=int, default=3)
    p.add_argument('--hole_override', type=str, default=None)

    # Logging
    p.add_argument('--log_every',       type=int, default=200)
    p.add_argument('--eval_every',      type=int, default=2000)
    p.add_argument('--eval_test_every', type=int, default=0)
    p.add_argument('--save_every',      type=int, default=20000)
    p.add_argument('--out_dir',         type=str, default='out')
    p.add_argument('--checkpoint', type=str, default=None)

    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    train(args)
