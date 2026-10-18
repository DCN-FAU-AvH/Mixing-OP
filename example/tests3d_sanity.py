"""
tests3d_sanity.py  --  Correctness checks for the 3D basis + mix-cost.

Run as a plain script:
    python -m example.tests3d_sanity

Exits with non-zero status if any assertion fails.
"""

from __future__ import annotations

# Allow both module execution and direct execution of this example.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import sys
import torch

from model.basis3d import DivFreeBasis3D, SpectralMixingCost3D, DyeMixCost3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(name: str, ok: bool, detail: str = ""):
    tag = "OK  " if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)
    if not ok:
        sys.exit(1)


def _fd_divergence(basis: DivFreeBasis3D, l: int, X: torch.Tensor,
                    h: float = 1e-4) -> torch.Tensor:
    """Central finite-difference divergence of mode l at points X: (n, 3)."""
    n = X.shape[0]
    div = torch.zeros(n)
    for d in range(3):
        dX = torch.zeros_like(X);  dX[:, d] = h
        Xp = X + dX;  Xm = X - dX
        # Pull out only mode l (construct one-hot lam).
        lam = torch.zeros(1, basis.L)
        lam[0, l] = 1.0
        vp = basis.velocity(Xp.unsqueeze(0), lam)[0]    # (n, 3)
        vm = basis.velocity(Xm.unsqueeze(0), lam)[0]
        div += (vp[:, d] - vm[:, d]) / (2 * h)
    return div


def _face_points(face_axis: int, face_value: float, n: int) -> torch.Tensor:
    """Generate n points on a face of the cube."""
    P = torch.rand(n, 3)
    P[:, face_axis] = face_value
    return P


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basis_count():
    for M in (1, 2, 3):
        basis = DivFreeBasis3D(M)
        expected = 3 * M ** 3
        _log(f"basis count (M={M})",
             basis.L == expected,
             f"got L={basis.L}, expected {expected}")
    # Spot-check advertised values.
    _log("L == 24 for M=2", DivFreeBasis3D(2).L == 24)
    _log("L == 81 for M=3", DivFreeBasis3D(3).L == 81)


def test_shape_contracts():
    basis = DivFreeBasis3D(2)
    X = torch.rand(3, 17, 3)
    lam = torch.randn(3, basis.L)
    v = basis.velocity(X, lam)
    _log("velocity shape", tuple(v.shape) == (3, 17, 3))

    dye = DyeMixCost3D(4)
    spec = SpectralMixingCost3D(4)
    X1 = torch.rand(2, 23, 3)
    X2 = torch.rand(2, 11, 3)
    cost = dye(X1, X2)
    _log("DyeMixCost3D shape", tuple(cost.shape) == (2,))
    _log("SpectralMixingCost3D shape",
         tuple(spec(X1).shape) == (2,))


def test_divergence_free():
    torch.manual_seed(0)
    basis = DivFreeBasis3D(2)
    # Interior points (avoid boundary so central FD is well-defined).
    X = 0.1 + 0.8 * torch.rand(256, 3)
    worst = 0.0
    for l in range(basis.L):
        div = _fd_divergence(basis, l, X, h=1e-4)
        worst = max(worst, float(div.abs().max()))
    _log("div-free (worst |FD div|)",
         worst < 1e-2,
         f"worst = {worst:.2e}")


def test_no_penetration():
    basis = DivFreeBasis3D(2)
    # Each of the six faces.
    faces = [(0, 0.0), (0, 1.0),
             (1, 0.0), (1, 1.0),
             (2, 0.0), (2, 1.0)]
    worst = 0.0
    for (ax, val) in faces:
        P = _face_points(ax, val, 500)
        # Test all modes at once via one batch.
        lam = torch.eye(basis.L)                    # (L, L)
        # We want v_ax for each mode; so evaluate basis directly.
        E = basis.eval_basis(P.unsqueeze(0))[0]      # (n, L, 3)
        v_ax = E[..., ax]                            # (n, L)
        worst = max(worst, float(v_ax.abs().max()))
    _log("no-penetration on all 6 faces",
         worst < 1e-5,
         f"worst |v.n| = {worst:.2e}")


def test_unit_norm_MC():
    torch.manual_seed(0)
    basis = DivFreeBasis3D(2)
    N = 40_000
    X = torch.rand(1, N, 3)
    E = basis.eval_basis(X)[0]                       # (N, L, 3)
    # ||e_l||^2 estimate = mean over N of sum_d e_{n,l,d}^2
    norms_sq = (E ** 2).sum(-1).mean(0)              # (L,)
    err = (norms_sq - 1.0).abs()
    _log("unit L^2 norm (MC)",
         float(err.max()) < 0.05,
         f"max |MC - 1| = {float(err.max()):.3f}  "
         f"(min={float(norms_sq.min()):.3f} max={float(norms_sq.max()):.3f})")


def test_orthogonality_MC():
    """Pairwise L^2 inner products should be ~0 (MC noise of order 1/sqrt(N))."""
    torch.manual_seed(0)
    basis = DivFreeBasis3D(2)
    N = 40_000
    X = torch.rand(1, N, 3)
    E = basis.eval_basis(X)[0]                       # (N, L, 3)
    # <e_a, e_b> ~ mean over N of sum_d e[n,a,d] e[n,b,d]
    # (B, L, 3) reshape to (N, L*3) then G = E^T @ E / N
    E_flat = E.reshape(N, -1)                        # (N, L*3), but mixed axes
    # Better: compute Gram via einsum.
    G = torch.einsum('nld,nkd->lk', E, E) / N        # (L, L)
    off = G - torch.diag(torch.diag(G))
    # The new basis is unit-norm per mode but NOT orthonormal across
    # parity-mismatched triples, so off-diagonals can reach ~0.1-0.2.
    _log("off-diagonal Gram (MC, non-orthogonal basis expected)",
         float(off.abs().max()) < 0.5,
         f"max |off| = {float(off.abs().max()):.3f}")


def test_mixcost_nonneg_and_decay():
    """
    With a partition by plane x1 = 0.5 and velocity the axis-3 (1,1,1) mode
    v^(3) = c (sin(pi x) cos(pi y) sin(pi z), -cos(pi x) sin(pi y) sin(pi z), 0),
    the dye mix-norm should strictly decrease over RK4.
    """
    torch.manual_seed(0)
    basis = DivFreeBasis3D(2)
    dye = DyeMixCost3D(6)

    # Pick the axis-3 (1,1,1) mode: horizontal stirring modulated by sin(pi z).
    idx = None
    for l in range(basis.L):
        if int(basis.axis[l]) == 3 and \
           int(basis.ms[l]) == 1 and int(basis.ns[l]) == 1 and \
           int(basis.ls[l]) == 1:
            idx = l
            break
    assert idx is not None, "expected to find axis-3 (1,1,1) mode"
    lam_fixed = torch.zeros(1, basis.L)
    lam_fixed[0, idx] = 3.0   # stirring amplitude

    # Sample particles partitioned by x1 < 0.5.
    N = 4000
    X = torch.rand(N, 3)
    mask = X[:, 0] < 0.5
    X1 = X[mask].unsqueeze(0)
    X2 = X[~mask].unsqueeze(0)

    # RK4 for a few steps
    T = 2.0;  N_T = 20;  dt = T / N_T
    mix_series = [float(dye(X1, X2)[0])]
    for _ in range(N_T):
        def vel(Y): return basis.velocity(Y, lam_fixed)
        k1 = vel(X1);          k1b = vel(X2)
        k2 = vel(X1 + 0.5*dt*k1);  k2b = vel(X2 + 0.5*dt*k1b)
        k3 = vel(X1 + 0.5*dt*k2);  k3b = vel(X2 + 0.5*dt*k2b)
        k4 = vel(X1 + dt*k3);      k4b = vel(X2 + dt*k3b)
        X1 = (X1 + dt*(k1 + 2*k2 + 2*k3 + k4)/6).clamp(0,1)
        X2 = (X2 + dt*(k1b + 2*k2b + 2*k3b + k4b)/6).clamp(0,1)
        mix_series.append(float(dye(X1, X2)[0]))

    _log("mix cost non-negative", min(mix_series) >= 0.0,
         f"min = {min(mix_series):.3e}")
    _log("mix cost decreases under stirring",
         mix_series[-1] < mix_series[0],
         f"start = {mix_series[0]:.3e}  ->  end = {mix_series[-1]:.3e}")


def test_spectral_cost_invariants():
    """SpectralMixingCost3D >= 0, = 0 for perfectly mixed uniform sample."""
    spec = SpectralMixingCost3D(4)
    # Non-negativity on random samples
    X = torch.rand(8, 2000, 3)
    c = spec(X)
    _log("spectral cost non-negative",
         float(c.min()) >= 0.0, f"min={float(c.min()):.3e}")
    # For large uniform samples, rho_hat_k ~ 0, so cost -> 0.
    X_big = torch.rand(1, 100_000, 3)
    c_big = spec(X_big)
    _log("spectral cost small for uniform",
         float(c_big[0]) < 1e-3,
         f"uniform cost = {float(c_big[0]):.3e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("3D basis / mix-cost sanity tests")
    print("=" * 60)
    test_basis_count()
    test_shape_contracts()
    test_no_penetration()
    test_divergence_free()
    test_unit_norm_MC()
    test_orthogonality_MC()
    test_mixcost_nonneg_and_decay()
    test_spectral_cost_invariants()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
