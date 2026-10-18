"""
train.py  --  Training script for two-density fluid mixing operator.

Problem (Lagrangian formulation)
--------------------------------
  min_{theta}  E_{partition} [
      lam_kin * int_0^T (1/N) sum_{i=1}^{N} ||v(t, X_i(t))||^2 dt
    + lam_mix * ||f(T)||^2_{H^{-1}(Omega)}
  ]

  where  N = n1 + n2 is the pooled trajectory cloud (uniform sample on Omega
         split by phase); the kinetic term is the Monte Carlo estimate of the
         plain kinetic energy int_0^T int_Omega ||v||^2 dx dt (volume-
         preserving flow, |Omega| = 1);
         f(T) uses centered signed weights w_tilde_i = w_i - mean(w),
         w_i = +1 for rho_1 particles, -1 for rho_2 particles (dye indicator)
         v(t,x) = sum_l lambda_l(t) * e_l(x)    (div-free basis expansion)
         lambda(t) = MixingOperatorNet({x^1_j}, {x^2_j}; t)

The operator takes TWO separate particle sets (rho_1 and rho_2) as context,
enabling it to produce different velocity fields for different spatial
configurations of the two densities.

Usage
-----
  python train.py --family rotation_line --lam_kin 2e-3
  python train.py --family circle --lam_kin 2e-3 --n_iter 100000 --h 512 --B_dist 16"""

from __future__ import annotations

import os
import math
import json
import argparse
import time

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched

from model.basis import DivFreeBasis, DyeMixCost
from model   import MixingOperatorNet
from model.dataset import build_family, make_partition_sampler
from model.util import setup_logger, log_run_header, log_training_summary


# ===========================================================================
# ODE Integrator
# ===========================================================================

def _lam_at(net: MixingOperatorNet, z: torch.Tensor,
            t_val: float, T: float) -> torch.Tensor:
    """Return lambda(t_val) for precomputed context z."""
    B = z.shape[0]
    t_norm = z.new_full((B,), t_val / T)
    return net.forward_from_ctx(z, t_norm)


def rk4_integrate(
    net:   MixingOperatorNet,
    basis: DivFreeBasis,
    z:     torch.Tensor,        # (B, h)
    X1_0:  torch.Tensor,        # (B, n1, 2)  rho_1 initial
    X2_0:  torch.Tensor,        # (B, n2, 2)  rho_2 initial
    mask1: torch.Tensor | None, # (B, n1)  True = real
    mask2: torch.Tensor | None, # (B, n2)  True = real
    T:     float = 1.0,
    N_T:   int   = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate BOTH particle sets with the SAME velocity field (RK4).

    Returns
    -------
    X1_T   : (B, n1, 2)  rho_1 terminal positions
    X2_T   : (B, n2, 2)  rho_2 terminal positions
    kin1   : (B,)         kinetic energy for rho_1
    kin2   : (B,)         kinetic energy for rho_2
    """
    dt  = T / N_T
    X1  = X1_0.clone()
    X2  = X2_0.clone()
    B   = X1.shape[0]
    kin1 = X1.new_zeros(B)
    kin2 = X2.new_zeros(B)

    # Precompute real particle counts for masked kinetic energy
    if mask1 is not None:
        n1_real = mask1.float().sum(-1).clamp(min=1)   # (B,)
    else:
        n1_real = torch.tensor(X1.shape[1], device=X1.device, dtype=X1.dtype)
    if mask2 is not None:
        n2_real = mask2.float().sum(-1).clamp(min=1)
    else:
        n2_real = torch.tensor(X2.shape[1], device=X2.device, dtype=X2.dtype)

    for s in range(N_T):
        t_s = s * dt

        # --- RK4 sub-steps: same lambda for both particle sets ---
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

        # Advance particles
        v1 = (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6.0
        v2 = (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6.0
        X1 = (X1 + dt * v1).clamp(0.0, 1.0)
        X2 = (X2 + dt * v2).clamp(0.0, 1.0)

        # Accumulate kinetic energy (midpoint velocity k2)
        # Mask out padded particles
        ke1 = (k2_1 ** 2).sum(-1)   # (B, n1)
        ke2 = (k2_2 ** 2).sum(-1)   # (B, n2)
        if mask1 is not None:
            ke1 = ke1 * mask1.float()
        if mask2 is not None:
            ke2 = ke2 * mask2.float()
        # Pooled normalization: E_kin = (dt/N) sum over ALL trajectory particles,
        # N = n1 + n2. Under uniform-split sampling this is the Monte Carlo
        # estimate of the plain kinetic energy int_0^T int_Omega |v|^2 dx dt
        # (volume-preserving flow). kin1/kin2 are the per-phase contributions.
        N_real = n1_real + n2_real
        kin1 += (ke1.sum(-1) / N_real) * dt
        kin2 += (ke2.sum(-1) / N_real) * dt

    return X1, X2, kin1, kin2


# ===========================================================================
# Loss
# ===========================================================================

def compute_loss(
    net:      MixingOperatorNet,
    basis:    DivFreeBasis,
    dye_cost: DyeMixCost,
    X1_ctx:   torch.Tensor,
    X2_ctx:   torch.Tensor,
    m1c:      torch.Tensor | None,
    m2c:      torch.Tensor | None,
    X1_traj:  torch.Tensor,
    X2_traj:  torch.Tensor,
    m1t:      torch.Tensor | None,
    m2t:      torch.Tensor | None,
    T:        float,
    N_T:      int,
    lam_mix:  float,
    lam_kin:  float,
) -> tuple[torch.Tensor, dict]:
    """Full forward pass: encode -> ODE -> loss."""
    # Encode context
    z = net.encode_context(X1_ctx, X2_ctx, m1c, m2c)

    # Integrate ODE
    X1_T, X2_T, kin1, kin2 = rk4_integrate(
        net, basis, z, X1_traj, X2_traj, m1t, m2t, T=T, N_T=N_T)

    # Terminal mixing cost
    mix_en = dye_cost(X1_T, X2_T, m1t, m2t)   # (B,)

    # Total loss
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
# Evaluation helper
# ===========================================================================

@torch.no_grad()
def evaluate(
    net:      MixingOperatorNet,
    basis:    DivFreeBasis,
    dye_cost: DyeMixCost,
    sampler,
    args,
    device: str,
    n_eval: int = 16,
) -> dict:
    """Run n_eval test problems and return average costs."""
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

    # --- Build partition family ---
    hole_override = json.loads(args.hole_override) if args.hole_override else None
    family = build_family(args.family, level_set_K=args.level_set_K,
                          hole_override=hole_override)
    if args.bounds_override:
        family.bounds = json.loads(args.bounds_override)
        print(f"Bounds overridden: {family.bounds}")

    # --- Default out_dir per family ---
    if args.out_dir == 'out':
        args.out_dir = os.path.join('out', f"{family.name}_d{family.dim}")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Set up logging ---
    logger = setup_logger(args.out_dir, name="train")
    train_start_time = time.time()

    # --- Build components ---
    basis    = DivFreeBasis(args.M_basis).to(device)
    dye_cost = DyeMixCost(args.K_mix).to(device)
    train_sampler = make_partition_sampler(family, mode='train')
    test_sampler  = make_partition_sampler(family, mode='test')

    L   = basis.L
    net = MixingOperatorNet(
        d              = 2,
        L              = L,
        h              = args.h,
        n_heads        = args.n_heads,
        n_layers_self  = args.n_layers_self,
        n_layers_joint = args.n_layers_joint,
        t_emb_dim      = args.t_emb_dim,
        dropout        = args.dropout,
    ).to(device)

    # --- Print run info ---
    log_run_header(logger, args, net, family=family)
    logger.info(f"  Div-free basis L   : {L}  (M_max={args.M_basis})")
    logger.info(f"  Spectral modes K   : {dye_cost.K}  (K_max={args.K_mix})")
    logger.info("")

    # --- Optimiser ---
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

    # --- Optional resume ---
    start_iter = 0
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt.get('iteration', 0)
        logger.info(f"Resumed from {args.checkpoint}  (iter {start_iter})")

    # --- Training loop ---
    net.train()
    log_loss = log_mix = log_kin1 = log_kin2 = 0.0
    t_start = time.time()

    for it in range(start_iter, args.n_iter):

        # Sample partitioned uniform particles (train mode: params outside hole)
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

        # --- Logging ---
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

        # --- Evaluation on train and test parameter distributions ---
        eval_interval = args.eval_test_every or args.eval_every
        if (it + 1) % eval_interval == 0:
            tr = evaluate(net, basis, dye_cost, train_sampler, args, device)
            te = evaluate(net, basis, dye_cost, test_sampler,  args, device)
            logger.info(
                f"  [eval train] mix={tr['mix']:.4f}  kin={tr['kin']:.4f}  |  "
                f"[eval test ] mix={te['mix']:.4f}  kin={te['kin']:.4f}"
            )

        # --- Checkpoint ---
        if (it + 1) % args.save_every == 0:
            path = os.path.join(args.out_dir, f"ckpt_{it+1:06d}.pt")
            torch.save({
                'model':     net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': it + 1,
                'args':      vars(args),
            }, path)
            logger.info(f"  [saved] {path}")

    # Final save
    path = os.path.join(args.out_dir, "ckpt_final.pt")
    torch.save({'model': net.state_dict(), 'args': vars(args)}, path)

    # --- Training summary ---
    log_training_summary(logger, train_start_time,
                         args.n_iter - start_iter, path)


# ===========================================================================
# CLI
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Train two-density fluid mixing operator")

    # Model
    p.add_argument('--M_basis',       type=int,   default=2,
                   help='Div-free basis wavenumber; L = M_basis^2')
    p.add_argument('--K_mix',         type=int,   default=12,
                   help='Spectral truncation for dye mix cost')
    p.add_argument('--h',             type=int,   default=256,
                   help='Transformer hidden dimension')
    p.add_argument('--n_heads',       type=int,   default=4,
                   help='Attention heads')
    p.add_argument('--n_layers_self', type=int,   default=1,
                   help='Self-attention depth per group')
    p.add_argument('--n_layers_joint',type=int,   default=1,
                   help='Joint self-attention depth')
    p.add_argument('--t_emb_dim',     type=int,   default=64,
                   help='Sinusoidal time embedding dim')
    p.add_argument('--dropout',       type=float, default=0.0)

    # ODE
    p.add_argument('--T',   type=float, default=5.0, help='Time horizon')
    p.add_argument('--N_T', type=int,   default=20,  help='RK4 time steps')

    # Data
    p.add_argument('--n_ctx',  type=int, default=256,
                   help='Total context particles (split by partition)')
    p.add_argument('--n_traj', type=int, default=256,
                   help='Total trajectory particles (split by partition)')
    p.add_argument('--B_dist', type=int, default=8,
                   help='Batch: number of distinct partitions per step')

    # Loss weights
    p.add_argument('--lam_mix', type=float, default=1.0,
                   help='Weight on dye mix-norm cost')
    p.add_argument('--lam_kin', type=float, default=1e-3,
                   help='Weight on kinetic energy')

    # Optimisation
    p.add_argument('--n_iter',    type=int,   default=100000)
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)

    # Partition family
    p.add_argument('--family', type=str, required=True,
                   choices=['rotation_line', 'general_line',
                            'circle', 'level_set'],
                   help='Parameterized partition family to train on')
    p.add_argument('--level_set_K', type=int, default=3,
                   help='Number of Fourier modes for level_set family')
    p.add_argument('--hole_override', type=str, default=None,
                   help='JSON list overriding the default test hole')
    p.add_argument('--bounds_override', type=str, default=None,
                   help='JSON list of [lo,hi] pairs overriding default bounds')

    # Logging / saving
    p.add_argument('--log_every',       type=int, default=200)
    p.add_argument('--eval_every',      type=int, default=2000)
    p.add_argument('--eval_test_every', type=int, default=0,
                   help='If >0, eval cadence for train/test split '
                        '(defaults to --eval_every)')
    p.add_argument('--save_every',      type=int, default=20000)
    p.add_argument('--out_dir',         type=str, default='out')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to checkpoint to resume from')

    # Device
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    train(args)
