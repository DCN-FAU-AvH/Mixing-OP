"""
basis3d.py  --  Div-free velocity basis and spectral mixing cost for [0,1]^3.

Div-free basis
--------------
On the cube Omega = (0,1)^3 we use the scalar potential

  psi_{m,n,l}(x) = sin(m pi x1) sin(n pi x2) sin(l pi x3)

curled around each Cartesian axis e_k, k in {1, 2, 3}:

  v^(1)_{m,n,l} = curl(psi e_x)
                = ( 0,
                     l sin(m pi x1) sin(n pi x2) cos(l pi x3),
                    -n sin(m pi x1) cos(n pi x2) sin(l pi x3) )

  v^(2)_{m,n,l} = curl(psi e_y)
                = ( -l sin(m pi x1) sin(n pi x2) cos(l pi x3),
                     0,
                     m cos(m pi x1) sin(n pi x2) sin(l pi x3) )

  v^(3)_{m,n,l} = curl(psi e_z)
                = ( n sin(m pi x1) cos(n pi x2) sin(l pi x3),
                   -m cos(m pi x1) sin(n pi x2) sin(l pi x3),
                    0 )

with m, n, l in {1, ..., M}.  Total basis size  L = 3 M^3   (L = 24 for M = 2).

Each mode is divergence-free (curl of anything is div-free) and satisfies
v . n = 0 on all six cube faces (each non-zero component has a sin factor
in its boundary-normal coordinate).

Unit-norm scaling (each mode normalised to ||e||_{L2((0,1)^3)} = 1) uses
the fact that one component is identically zero and sin^2 / cos^2 each
integrate to 1/2 on [0,1]:

  axis k=1:  c = 2 sqrt(2) / sqrt(n^2 + l^2)
  axis k=2:  c = 2 sqrt(2) / sqrt(m^2 + l^2)
  axis k=3:  c = 2 sqrt(2) / sqrt(m^2 + n^2)

Structurally this matches the 2D formula c_{mn} = 2 / sqrt(m^2 + n^2),
scaled by sqrt(2) from the extra Cartesian axis.

Caveats:
  * The basis is NOT complete: axis-invariant flows (any index = 0) are
    excluded by construction since sin(0) = 0.  This is acceptable for the
    fully three-dimensional partition families used here.
  * Modes are unit-norm individually but NOT orthonormal across triples
    with mismatched parities.  The kinetic-energy penalty is an MC
    estimator of int ||v||^2 dx at particle positions, so it is insensitive
    to basis orthogonality.

Spectral mixing cost
--------------------
H^{-1} norm on [0,1]^3 via Neumann cosine eigenfunctions

  rho_hat_k = (1/n) sum_i a_{k1} a_{k2} a_{k3} cos(k1 pi x1_i) cos(k2 pi x2_i)
                                                cos(k3 pi x3_i)

  lambda_k  = pi^2 (k1^2 + k2^2 + k3^2)

  ||rho||^2_{H^{-1}}  =  sum_{k != 0}  rho_hat_k^2 / lambda_k
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Div-free basis
# ---------------------------------------------------------------------------

class DivFreeBasis3D(nn.Module):
    """
    Div-free trigonometric basis on [0,1]^3 built as curls of the scalar
    potential  psi(x) = sin(m pi x1) sin(n pi x2) sin(l pi x3)  around
    each of the three Cartesian axes.

    Parameters
    ----------
    M_max : int
        Maximum wavenumber.  Basis size is L = 3 M_max^3.
    """

    def __init__(self, M_max: int):
        super().__init__()
        self.M_max = M_max

        # Enumerate (axis, m, n, l), axis in {1, 2, 3}, m,n,l in {1..M}.
        idx = [(k, m, n, l)
               for k in (1, 2, 3)
               for m in range(1, M_max + 1)
               for n in range(1, M_max + 1)
               for l in range(1, M_max + 1)]
        axis, ms, ns, ls = zip(*idx)
        axis_t = torch.tensor(axis, dtype=torch.int64)
        ms_t   = torch.tensor(ms,   dtype=torch.int64)
        ns_t   = torch.tensor(ns,   dtype=torch.int64)
        ls_t   = torch.tensor(ls,   dtype=torch.int64)

        # Per-mode L2([0,1]^3) normalisation.  For axis k the non-axial
        # index pair contributes (i^2 + j^2) / 8 to ||e||^2, hence
        # c = 2 sqrt(2) / sqrt(i^2 + j^2).
        pair_sq = torch.where(axis_t == 1, ns_t ** 2 + ls_t ** 2,
                   torch.where(axis_t == 2, ms_t ** 2 + ls_t ** 2,
                                            ms_t ** 2 + ns_t ** 2)).float()
        norm = (2.0 * math.sqrt(2.0)) / torch.sqrt(pair_sq)

        self.L = len(idx)
        assert self.L == 3 * M_max ** 3, (
            f"bad L: got {self.L}, expected {3 * M_max ** 3}")

        self.register_buffer('axis', axis_t)          # 1, 2, 3
        self.register_buffer('ms',   ms_t)
        self.register_buffer('ns',   ns_t)
        self.register_buffer('ls',   ls_t)
        self.register_buffer('norm', norm)            # (L,)

    # ------------------------------------------------------------------
    def eval_basis(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all basis vectors at a set of points.

        Parameters
        ----------
        X : (B, n, 3)  or  (n, 3)

        Returns
        -------
        E : (..., n, L, 3)  --  basis vectors at each point
        """
        squeeze = (X.dim() == 2)
        if squeeze:
            X = X.unsqueeze(0)              # (1, n, 3)

        B, n, d = X.shape
        assert d == 3, f"X must have spatial dim 3, got {d}"

        x1 = X[..., 0:1]                    # (B, n, 1)
        x2 = X[..., 1:2]
        x3 = X[..., 2:3]

        m  = self.ms.view(1, 1, -1).to(X.dtype)    # (1, 1, L)
        nv = self.ns.view(1, 1, -1).to(X.dtype)
        lv = self.ls.view(1, 1, -1).to(X.dtype)

        pi = math.pi
        sm = torch.sin(m  * pi * x1);  cm = torch.cos(m  * pi * x1)
        sn = torch.sin(nv * pi * x2);  cn = torch.cos(nv * pi * x2)
        sl = torch.sin(lv * pi * x3);  cl = torch.cos(lv * pi * x3)

        # Three shared products (each appears in two of the three axis families).
        P_ssc = sm * sn * cl        # v^(1)_y, -v^(2)_x
        P_scs = sm * cn * sl        # -v^(1)_z,  v^(3)_x
        P_css = cm * sn * sl        # v^(2)_z, -v^(3)_y

        axis = self.axis.view(1, 1, -1)
        m1 = (axis == 1).to(X.dtype)
        m2 = (axis == 2).to(X.dtype)
        m3 = (axis == 3).to(X.dtype)

        # Assemble unscaled components (unit-norm scaling applied last).
        ex = (nv * P_scs) * m3 - (lv * P_ssc) * m2
        ey = (lv * P_ssc) * m1 - (m  * P_css) * m3
        ez = (m  * P_css) * m2 - (nv * P_scs) * m1

        norm = self.norm.view(1, 1, -1)
        ex = ex * norm
        ey = ey * norm
        ez = ez * norm

        E = torch.stack([ex, ey, ez], dim=-1)   # (B, n, L, 3)
        return E.squeeze(0) if squeeze else E

    # ------------------------------------------------------------------
    def velocity(self, X: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        Compute  v(x) = sum_l  lam_l * e_l(x).

        Parameters
        ----------
        X   : (B, n, 3)
        lam : (B, L)

        Returns
        -------
        v : (B, n, 3)
        """
        E = self.eval_basis(X)                          # (B, n, L, 3)
        return torch.einsum('bl,bnld->bnd', lam, E)     # (B, n, 3)


# ---------------------------------------------------------------------------
# Helpers for 3D spectral costs
# ---------------------------------------------------------------------------

def _build_neumann_triples(K_max: int):
    """Enumerate (k1, k2, k3) in {0..K}^3 excluding (0,0,0), with Neumann weights."""
    triples = [(k1, k2, k3)
               for k1 in range(K_max + 1)
               for k2 in range(K_max + 1)
               for k3 in range(K_max + 1)
               if not (k1 == 0 and k2 == 0 and k3 == 0)]
    k1s, k2s, k3s = zip(*triples)
    k1s = torch.tensor(k1s, dtype=torch.float32)
    k2s = torch.tensor(k2s, dtype=torch.float32)
    k3s = torch.tensor(k3s, dtype=torch.float32)

    eigs = math.pi ** 2 * (k1s ** 2 + k2s ** 2 + k3s ** 2)

    def _a(k: torch.Tensor) -> torch.Tensor:
        return torch.where(k == 0,
                           torch.ones_like(k),
                           math.sqrt(2.0) * torch.ones_like(k))
    ak = _a(k1s) * _a(k2s) * _a(k3s)
    return k1s, k2s, k3s, eigs, ak


# ---------------------------------------------------------------------------
# Spectral mixing cost
# ---------------------------------------------------------------------------

class SpectralMixingCost3D(nn.Module):
    """
    Estimates  ||rho(T)||^2_{H^{-1}([0,1]^3)}  from particle positions.

    Parameters
    ----------
    K_max : int
        Truncation: include modes (k1,k2,k3) in {0,...,K_max}^3 except (0,0,0).
    """

    def __init__(self, K_max: int):
        super().__init__()
        self.K_max = K_max
        k1s, k2s, k3s, eigs, ak = _build_neumann_triples(K_max)
        self.register_buffer('k1s',  k1s)
        self.register_buffer('k2s',  k2s)
        self.register_buffer('k3s',  k3s)
        self.register_buffer('eigs', eigs)
        self.register_buffer('ak',   ak)
        self.K = len(k1s)

    def forward(self, X_T: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X_T : (B, n, 3)   terminal particle positions in [0,1]^3

        Returns
        -------
        cost : (B,)   mixing cost  (smaller = better mixed)
        """
        x1 = X_T[..., 0]
        x2 = X_T[..., 1]
        x3 = X_T[..., 2]
        pi = math.pi

        k1 = self.k1s.view(1, 1, -1)
        k2 = self.k2s.view(1, 1, -1)
        k3 = self.k3s.view(1, 1, -1)
        ak = self.ak.view(1, 1, -1)
        eigs = self.eigs.view(1, -1)

        phi = ak * torch.cos(k1 * pi * x1.unsqueeze(-1)) \
                 * torch.cos(k2 * pi * x2.unsqueeze(-1)) \
                 * torch.cos(k3 * pi * x3.unsqueeze(-1))     # (B, n, K)

        rho_hat = phi.mean(dim=1)                             # (B, K)
        return (rho_hat ** 2 / eigs).sum(-1)                  # (B,)


# ---------------------------------------------------------------------------
# Dye mixing cost
# ---------------------------------------------------------------------------

class DyeMixCost3D(nn.Module):
    """
    Batched dye-mixing cost  ||f||^2_{H^{-1}([0,1]^3)}
    with  f_i = +1  for rho_1 particles  and  f_i = -1  for rho_2 particles.

    Parameters
    ----------
    K_max : int
        Spectral truncation over {0..K_max}^3 except (0,0,0).
    """

    def __init__(self, K_max: int):
        super().__init__()
        self.K_max = K_max
        k1s, k2s, k3s, eigs, ak = _build_neumann_triples(K_max)
        self.register_buffer('k1s',  k1s)
        self.register_buffer('k2s',  k2s)
        self.register_buffer('k3s',  k3s)
        self.register_buffer('eigs', eigs)
        self.register_buffer('ak',   ak)
        self.K = len(k1s)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor,
                mask1: torch.Tensor | None = None,
                mask2: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        X1    : (B, n1, 3)  rho_1 terminal positions
        X2    : (B, n2, 3)  rho_2 terminal positions
        mask1 : (B, n1)     True for real particles (optional)
        mask2 : (B, n2)     True for real particles (optional)

        Returns
        -------
        cost : (B,)   dye mix-norm  (smaller = better mixed)
        """
        pi = math.pi
        k1 = self.k1s.view(1, 1, -1)
        k2 = self.k2s.view(1, 1, -1)
        k3 = self.k3s.view(1, 1, -1)
        ak = self.ak.view(1, 1, -1)
        eigs = self.eigs.view(1, -1)

        def _eval_phi(X):
            x1 = X[..., 0]
            x2 = X[..., 1]
            x3 = X[..., 2]
            return ak * torch.cos(k1 * pi * x1.unsqueeze(-1)) \
                      * torch.cos(k2 * pi * x2.unsqueeze(-1)) \
                      * torch.cos(k3 * pi * x3.unsqueeze(-1))    # (B, n, K)

        phi1 = _eval_phi(X1)
        phi2 = _eval_phi(X2)

        if mask1 is not None:
            phi1 = phi1 * mask1.unsqueeze(-1).float()
            n1 = mask1.float().sum(-1, keepdim=True)             # (B, 1)
        else:
            n1 = torch.tensor(X1.shape[1], device=X1.device, dtype=X1.dtype)
        if mask2 is not None:
            phi2 = phi2 * mask2.unsqueeze(-1).float()
            n2 = mask2.float().sum(-1, keepdim=True)
        else:
            n2 = torch.tensor(X2.shape[1], device=X1.device, dtype=X1.dtype)

        N_total = n1 + n2
        sum1 = phi1.sum(dim=1)                                   # (B, K)
        sum2 = phi2.sum(dim=1)
        # Centered signed weights: w_i = +1 (rho_1) / -1 (rho_2),
        # w_tilde_i = w_i - w_bar with w_bar = (n1 - n2)/N, so that the signed
        # empirical measure has exact zero mean (particle analogue of P_0 rho).
        # f_hat_k = (1/N) [ (sum1 - sum2) - w_bar (sum1 + sum2) ]
        w_bar = (n1 - n2) / N_total
        f_hat = ((sum1 - sum2) - w_bar * (sum1 + sum2)) / N_total   # (B, K)

        return (f_hat ** 2 / eigs).sum(-1)                       # (B,)
