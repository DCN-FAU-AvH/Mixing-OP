"""
dataset.py  --  Parameterized partition families on [0,1]^2.

Each training run targets a SINGLE partition family with known manifold
dimension.  Parameters are sampled from the complement of a "test hole"
at training time, and from inside the hole at evaluation time, so that
generalization can be measured cleanly (interval hole-out).

Families
--------
  rotation_line : angle theta                     (1D manifold)
  general_line  : (theta, offset)                 (2D manifold)
  circle        : (cx, cy, r)                     (3D manifold)
  level_set     : Fourier-series coefficients     (2K-dim manifold)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PartitionFamily:
    """
    Abstract base.  Subclasses define ``bounds``, ``hole`` and the
    ``predicate`` method.
    """

    name: str = "base"
    dim:  int = 0
    # Full manifold bounds per dimension: list of (lo, hi).
    bounds: List[Tuple[float, float]] = []
    # Test hole per dimension: list of Optional[(lo, hi)].  ``None`` on a
    # dimension means "no hole" (that coordinate is free for both modes).
    hole:   List[Optional[Tuple[float, float]]] = []

    # ---- Sampling --------------------------------------------------------
    def _uniform_in_bounds(self, device) -> Tensor:
        p = torch.empty(self.dim, device=device)
        for i, (lo, hi) in enumerate(self.bounds):
            p[i] = torch.rand((), device=device) * (hi - lo) + lo
        return p

    def _in_hole(self, p: Tensor) -> bool:
        for i, h in enumerate(self.hole):
            if h is None:
                continue
            lo, hi = h
            if not (lo <= float(p[i]) <= hi):
                return False
        return True

    def _sample_in_hole(self, device) -> Tensor:
        p = torch.empty(self.dim, device=device)
        for i, (lo, hi) in enumerate(self.bounds):
            h = self.hole[i]
            if h is not None:
                lo, hi = h
            p[i] = torch.rand((), device=device) * (hi - lo) + lo
        return p

    def sample_param(self, mode: str = "train", device: str = "cpu",
                     max_tries: int = 200) -> Tensor:
        """
        Draw one parameter.

          mode='train' : rejection-sample in ``bounds \\ hole``.
          mode='test'  : uniform inside ``hole`` (across held-out dims;
                         free dims still span their full bounds).
        """
        if mode == "test":
            for _ in range(max_tries):
                p = self._sample_in_hole(device)
                if self._valid(p):
                    return p
            raise RuntimeError(
                f"{self.name}: could not sample a valid 'test' point")
        # train: reject anything inside the hole box
        for _ in range(max_tries):
            p = self._uniform_in_bounds(device)
            if not self._in_hole(p) and self._valid(p):
                return p
        raise RuntimeError(
            f"{self.name}: could not sample a valid 'train' point")

    # ---- Validation hook -------------------------------------------------
    def _valid(self, p: Tensor) -> bool:
        """
        Override in subclasses that need to reject degenerate parameters
        (e.g. level_set with one empty class).
        """
        return True

    # ---- Predicate -------------------------------------------------------
    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        raise NotImplementedError

    # ---- Description -----------------------------------------------------
    def describe(self, p: Tensor) -> str:
        vals = ", ".join(f"{float(v):.3f}" for v in p)
        return f"{self.name}({vals})"

    # ---- Serialisation ---------------------------------------------------
    def info(self) -> Dict:
        """Human-readable summary, dumped into logs and figures."""
        return {
            "name":   self.name,
            "dim":    self.dim,
            "bounds": [list(b) for b in self.bounds],
            "hole":   [list(h) if h is not None else None for h in self.hole],
        }


# ---------------------------------------------------------------------------
# 1D — rotation line through the domain centre
# ---------------------------------------------------------------------------

class RotationLineFamily(PartitionFamily):
    name = "rotation_line"
    dim  = 1
    bounds = [(0.0, math.pi)]
    hole   = [(math.pi / 3.0, 2.0 * math.pi / 3.0)]

    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        theta = p[0]
        n = torch.stack([torch.cos(theta), torch.sin(theta)])      # (2,)
        return ((X - 0.5) * n).sum(-1) < 0.0


# ---------------------------------------------------------------------------
# 2D — general line (angle + signed offset)
# ---------------------------------------------------------------------------

class GeneralLineFamily(PartitionFamily):
    name = "general_line"
    dim  = 2
    bounds = [(0.0, math.pi), (-0.3, 0.3)]
    hole   = [(math.pi / 3.0, 2.0 * math.pi / 3.0), (-0.1, 0.1)]

    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        theta, b = p[0], p[1]
        n = torch.stack([torch.cos(theta), torch.sin(theta)])      # (2,)
        return ((X - 0.5) * n).sum(-1) < b


# ---------------------------------------------------------------------------
# 3D — random disc (centre + radius)
# ---------------------------------------------------------------------------

class CircleFamily(PartitionFamily):
    name = "circle"
    dim  = 3
    bounds = [(0.25, 0.75), (0.25, 0.75), (0.15, 0.35)]
    hole   = [(0.40, 0.60), (0.40, 0.60), (0.20, 0.30)]

    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        cx, cy, r = p[0], p[1], p[2]
        centre = torch.stack([cx, cy])
        dist2  = ((X - centre) ** 2).sum(-1)
        return dist2 < r * r


# ---------------------------------------------------------------------------
# 2K-D — Fourier-series level set
# ---------------------------------------------------------------------------

class LevelSetFamily(PartitionFamily):
    """
    f(x1, x2) = sum_{k=1}^K  a_k sin(k*pi*x1) cos(k*pi*x2)
                           + b_k cos(k*pi*x1) sin(k*pi*x2)

    rho_1 iff f > 0.  Parameters stacked as (a_1..a_K, b_1..b_K).
    """
    name = "level_set"

    def __init__(self, K: int = 3):
        self.K   = K
        self.dim = 2 * K
        # Each coefficient in [-1, 1].
        self.bounds = [(-1.0, 1.0)] * self.dim
        # Default hole: restrict a_1 only.
        self.hole = [None] * self.dim
        self.hole[0] = (-0.3, 0.3)

    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        K = self.K
        a = p[:K]           # (K,)
        b = p[K:2 * K]      # (K,)
        x1 = X[..., 0]      # (n,)
        x2 = X[..., 1]
        pi = math.pi

        ks = torch.arange(1, K + 1, device=X.device, dtype=X.dtype)      # (K,)
        arg_x1 = ks[None, :] * pi * x1[:, None]                          # (n, K)
        arg_x2 = ks[None, :] * pi * x2[:, None]
        term_a = torch.sin(arg_x1) * torch.cos(arg_x2)                   # (n, K)
        term_b = torch.cos(arg_x1) * torch.sin(arg_x2)                   # (n, K)
        f = (term_a * a[None, :]).sum(-1) + (term_b * b[None, :]).sum(-1)
        return f > 0.0

    def _valid(self, p: Tensor) -> bool:
        """Reject params that give an almost-empty class (< 20% of probes)."""
        X = torch.rand(256, 2, device=p.device)
        mask = self.predicate(X, p)
        frac = float(mask.float().mean())
        return 0.2 <= frac <= 0.8


# ---------------------------------------------------------------------------
# Registry + builder
# ---------------------------------------------------------------------------

PARTITION_FAMILIES = {
    "rotation_line": RotationLineFamily,
    "general_line":  GeneralLineFamily,
    "circle":        CircleFamily,
    "level_set":     LevelSetFamily,
}


def build_family(name: str, level_set_K: int = 3,
                 hole_override: Optional[List[Optional[Tuple[float, float]]]]
                 = None) -> PartitionFamily:
    if name not in PARTITION_FAMILIES:
        raise ValueError(
            f"Unknown family '{name}'. "
            f"Choose from {list(PARTITION_FAMILIES)}.")
    cls = PARTITION_FAMILIES[name]
    fam = cls(K=level_set_K) if name == "level_set" else cls()
    if hole_override is not None:
        if len(hole_override) != fam.dim:
            raise ValueError(
                f"hole_override has length {len(hole_override)}, "
                f"family dim is {fam.dim}")
        fam.hole = list(hole_override)
    return fam


# ---------------------------------------------------------------------------
# Padding helper (unchanged mechanics)
# ---------------------------------------------------------------------------

def _pad_batch(tensors: List[Tensor], device: str) -> Tuple[Tensor, Tensor]:
    max_n = max((t.shape[0] for t in tensors), default=0)
    if max_n == 0:
        max_n = 1
    B = len(tensors)
    X_pad = torch.zeros(B, max_n, 2, device=device)
    mask  = torch.zeros(B, max_n, dtype=torch.bool, device=device)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        if n > 0:
            X_pad[i, :n] = t
            mask[i, :n] = True
    return X_pad, mask


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------

def make_partition_sampler(family: PartitionFamily, mode: str = "train"):
    """
    Returns ``sampler(B, n_ctx, n_traj, device)`` that produces

        X1_ctx, X2_ctx, m1c, m2c, X1_traj, X2_traj, m1t, m2t

    with per-element parameters drawn from ``family.sample_param(mode)``.
    """
    if mode not in ("train", "test"):
        raise ValueError(f"mode must be 'train' or 'test', got {mode}")

    def sampler(B: int, n_ctx: int, n_traj: int, device: str = "cpu"):
        ctx_1, ctx_2, traj_1, traj_2 = [], [], [], []
        for _ in range(B):
            p = family.sample_param(mode, device)
            X_all    = torch.rand(n_ctx + n_traj, 2, device=device)
            mask_all = family.predicate(X_all, p)

            X_ctx  = X_all[:n_ctx]
            m_ctx  = mask_all[:n_ctx]
            X_traj = X_all[n_ctx:]
            m_traj = mask_all[n_ctx:]

            ctx_1.append(X_ctx[m_ctx])
            ctx_2.append(X_ctx[~m_ctx])
            traj_1.append(X_traj[m_traj])
            traj_2.append(X_traj[~m_traj])

        X1c, m1c = _pad_batch(ctx_1,  device)
        X2c, m2c = _pad_batch(ctx_2,  device)
        X1t, m1t = _pad_batch(traj_1, device)
        X2t, m2t = _pad_batch(traj_2, device)
        return X1c, X2c, m1c, m2c, X1t, X2t, m1t, m2t

    return sampler


# ---------------------------------------------------------------------------
# Named partition predicates used by example/test.py
# ---------------------------------------------------------------------------

def get_named_partition(name: str):
    """
    Backward-compatible partition predicate by name.  Returns a function
    ``f(X) -> BoolTensor`` suitable for the existing example/test.py.
    """
    fam_line = RotationLineFamily()
    if name == "left_right":
        theta = torch.tensor([0.0])
        return lambda X: fam_line.predicate(X, theta)
    if name == "top_bottom":
        theta = torch.tensor([math.pi / 2.0])
        return lambda X: fam_line.predicate(X, theta)
    if name == "diagonal":
        theta = torch.tensor([math.pi / 4.0])
        return lambda X: fam_line.predicate(X, theta)
    if name == "radial":
        fam_c = CircleFamily()
        p = torch.tensor([0.5, 0.5, 0.25])
        return lambda X: fam_c.predicate(X, p)
    raise ValueError(f"Unknown legacy partition '{name}'.")
