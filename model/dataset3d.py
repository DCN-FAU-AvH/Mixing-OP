"""
dataset3d.py  --  Parameterized partition families on [0,1]^3.

3D analogue of dataset.py.  Each training run targets a single family with
known manifold dimension.  Parameters are drawn from the complement of a
"test hole" at training time, and from inside the hole at evaluation time
(interval hole-out).

Families
--------
  rotation_plane : unit-sphere orientation  (theta, phi)       (2D manifold)
  general_plane  : orientation + signed offset                 (3D manifold)
  sphere         : centre in [0.25,0.75]^3 + radius            (4D manifold)
  level_set_3d   : Fourier-series coefficients                 (3K-D manifold)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Base class (mirrors dataset.PartitionFamily)
# ---------------------------------------------------------------------------

class PartitionFamily3D:
    """
    Abstract base for 3D partitions on [0,1]^3.  Subclasses define
    ``bounds``, ``hole`` and ``predicate``.
    """

    name: str = "base3d"
    dim:  int = 0
    bounds: List[Tuple[float, float]] = []
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
                     max_tries: int = 400) -> Tensor:
        if mode == "test":
            for _ in range(max_tries):
                p = self._sample_in_hole(device)
                if self._valid(p):
                    return p
            raise RuntimeError(
                f"{self.name}: could not sample a valid 'test' point")
        for _ in range(max_tries):
            p = self._uniform_in_bounds(device)
            if not self._in_hole(p) and self._valid(p):
                return p
        raise RuntimeError(
            f"{self.name}: could not sample a valid 'train' point")

    def _valid(self, p: Tensor) -> bool:
        return True

    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        raise NotImplementedError

    def describe(self, p: Tensor) -> str:
        vals = ", ".join(f"{float(v):.3f}" for v in p)
        return f"{self.name}({vals})"

    def info(self) -> Dict:
        return {
            "name":   self.name,
            "dim":    self.dim,
            "bounds": [list(b) for b in self.bounds],
            "hole":   [list(h) if h is not None else None for h in self.hole],
        }


# ---------------------------------------------------------------------------
# Unit-normal helper  (theta in [0,pi], phi in [0,pi] covers S^2 / {+/-})
# ---------------------------------------------------------------------------

def _unit_normal(theta: Tensor, phi: Tensor) -> Tensor:
    """Return n = (sin phi cos theta, sin phi sin theta, cos phi)."""
    return torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi),
    ])


# ---------------------------------------------------------------------------
# 2D — plane through centre (orientation on S^2, no offset)
# ---------------------------------------------------------------------------

class RotationPlaneFamily(PartitionFamily3D):
    name = "rotation_plane"
    dim  = 2
    # theta in [0, pi], phi in [0, pi].  Covers all unbased half-spaces.
    bounds = [(0.0, math.pi), (0.0, math.pi)]
    # Test hole: middle third in each angle.
    hole   = [(math.pi / 3.0, 2.0 * math.pi / 3.0),
              (math.pi / 3.0, 2.0 * math.pi / 3.0)]

    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        n = _unit_normal(p[0], p[1])          # (3,)
        return ((X - 0.5) * n).sum(-1) < 0.0


# ---------------------------------------------------------------------------
# 3D — general plane (orientation + signed offset)
# ---------------------------------------------------------------------------

class GeneralPlaneFamily(PartitionFamily3D):
    name = "general_plane"
    dim  = 3
    bounds = [(0.0, math.pi), (0.0, math.pi), (-0.3, 0.3)]
    hole   = [(math.pi / 3.0, 2.0 * math.pi / 3.0),
              (math.pi / 3.0, 2.0 * math.pi / 3.0),
              (-0.1, 0.1)]

    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        n = _unit_normal(p[0], p[1])
        b = p[2]
        return ((X - 0.5) * n).sum(-1) < b


# ---------------------------------------------------------------------------
# 4D — ball (centre + radius)
# ---------------------------------------------------------------------------

class SphereFamily(PartitionFamily3D):
    name = "sphere"
    dim  = 4
    bounds = [(0.40, 0.60), (0.40, 0.60), (0.40, 0.60), (0.42, 0.54)]
    hole   = [(0.45, 0.55), (0.45, 0.55), (0.45, 0.55), (0.48, 0.52)]

    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        centre = torch.stack([p[0], p[1], p[2]])
        r = p[3]
        return ((X - centre) ** 2).sum(-1) < r * r


# ---------------------------------------------------------------------------
# 3K-D — Fourier-series level set on [0,1]^3
# ---------------------------------------------------------------------------

class LevelSet3DFamily(PartitionFamily3D):
    """
    f(x1,x2,x3) = sum_{k=1}^K
        a_k sin(k pi x1) cos(k pi x2) cos(k pi x3)
      + b_k cos(k pi x1) sin(k pi x2) cos(k pi x3)
      + c_k cos(k pi x1) cos(k pi x2) sin(k pi x3)

    rho_1 iff f > 0.  Parameters are (a_1..a_K, b_1..b_K, c_1..c_K).
    """
    name = "level_set_3d"

    def __init__(self, K: int = 3):
        self.K   = K
        self.dim = 3 * K
        self.bounds = [(-1.0, 1.0)] * self.dim
        self.hole = [None] * self.dim
        # Restrict a_1 as a non-trivial hole (mirrors 2D level_set).
        self.hole[0] = (-0.3, 0.3)

    def predicate(self, X: Tensor, p: Tensor) -> Tensor:
        K = self.K
        a = p[      :  K]
        b = p[  K   : 2*K]
        c = p[2*K   : 3*K]
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        pi = math.pi

        ks = torch.arange(1, K + 1, device=X.device, dtype=X.dtype)    # (K,)
        ax1 = ks[None, :] * pi * x1[:, None]                            # (n, K)
        ax2 = ks[None, :] * pi * x2[:, None]
        ax3 = ks[None, :] * pi * x3[:, None]
        s1 = torch.sin(ax1);  c1 = torch.cos(ax1)
        s2 = torch.sin(ax2);  c2 = torch.cos(ax2)
        s3 = torch.sin(ax3);  c3 = torch.cos(ax3)

        term_a = s1 * c2 * c3
        term_b = c1 * s2 * c3
        term_c = c1 * c2 * s3
        f = (term_a * a[None, :]).sum(-1) \
          + (term_b * b[None, :]).sum(-1) \
          + (term_c * c[None, :]).sum(-1)
        return f > 0.0

    def _valid(self, p: Tensor) -> bool:
        """Reject degenerate parameters (<20% or >80% of one class)."""
        X = torch.rand(512, 3, device=p.device)
        mask = self.predicate(X, p)
        frac = float(mask.float().mean())
        return 0.2 <= frac <= 0.8


# ---------------------------------------------------------------------------
# Registry + builder
# ---------------------------------------------------------------------------

PARTITION_FAMILIES_3D = {
    "rotation_plane": RotationPlaneFamily,
    "general_plane":  GeneralPlaneFamily,
    "sphere":         SphereFamily,
    "level_set_3d":   LevelSet3DFamily,
}


def build_family_3d(name: str, level_set_K: int = 3,
                    hole_override: Optional[List[Optional[Tuple[float, float]]]]
                    = None) -> PartitionFamily3D:
    if name not in PARTITION_FAMILIES_3D:
        raise ValueError(
            f"Unknown 3D family '{name}'. "
            f"Choose from {list(PARTITION_FAMILIES_3D)}.")
    cls = PARTITION_FAMILIES_3D[name]
    fam = cls(K=level_set_K) if name == "level_set_3d" else cls()
    if hole_override is not None:
        if len(hole_override) != fam.dim:
            raise ValueError(
                f"hole_override has length {len(hole_override)}, "
                f"family dim is {fam.dim}")
        fam.hole = list(hole_override)
    return fam


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------

def _pad_batch_3d(tensors: List[Tensor], device: str) -> Tuple[Tensor, Tensor]:
    max_n = max((t.shape[0] for t in tensors), default=0)
    if max_n == 0:
        max_n = 1
    B = len(tensors)
    X_pad = torch.zeros(B, max_n, 3, device=device)
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

def make_partition_sampler_3d(family: PartitionFamily3D, mode: str = "train"):
    """
    Returns ``sampler(B, n_ctx, n_traj, device)`` producing

        X1_ctx, X2_ctx, m1c, m2c, X1_traj, X2_traj, m1t, m2t

    Per-element parameters are drawn from ``family.sample_param(mode)``.
    """
    if mode not in ("train", "test"):
        raise ValueError(f"mode must be 'train' or 'test', got {mode}")

    def sampler(B: int, n_ctx: int, n_traj: int, device: str = "cpu"):
        ctx_1, ctx_2, traj_1, traj_2 = [], [], [], []
        for _ in range(B):
            p = family.sample_param(mode, device)
            X_all    = torch.rand(n_ctx + n_traj, 3, device=device)
            mask_all = family.predicate(X_all, p)

            X_ctx  = X_all[:n_ctx]
            m_ctx  = mask_all[:n_ctx]
            X_traj = X_all[n_ctx:]
            m_traj = mask_all[n_ctx:]

            ctx_1.append(X_ctx[m_ctx])
            ctx_2.append(X_ctx[~m_ctx])
            traj_1.append(X_traj[m_traj])
            traj_2.append(X_traj[~m_traj])

        X1c, m1c = _pad_batch_3d(ctx_1,  device)
        X2c, m2c = _pad_batch_3d(ctx_2,  device)
        X1t, m1t = _pad_batch_3d(traj_1, device)
        X2t, m2t = _pad_batch_3d(traj_2, device)
        return X1c, X2c, m1c, m2c, X1t, X2t, m1t, m2t

    return sampler
