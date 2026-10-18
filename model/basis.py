"""
basis.py  –  Div-free velocity basis and spectral mixing cost for [0,1]^2.

Div-free basis
--------------
Derived from stream functions  psi_{mn}(x) = sin(m pi x1) sin(n pi x2),  m,n >= 1.

  e_{mn}(x) = nabla^perp psi_{mn}
             = ( d/dx2 psi_{mn},  -d/dx1 psi_{mn} )
             = ( n pi sin(m pi x1) cos(n pi x2),
                -m pi cos(m pi x1) sin(n pi x2) )

Properties (all hold exactly):
  div(e_{mn}) = 0              (incompressible)
  e_{mn} . n  = 0  on dOmega  (no normal flow)

The basis is normalised so that ||e_{mn}||_{L2([0,1]^2)} = 1
  (each component has L2 norm  (pi/2) sqrt(m^2+n^2) / 2,
   so the full L2 norm squared is  pi^2(m^2+n^2)/4, divide by that).

Spectral mixing cost
--------------------
H^{-1} norm on [0,1]^2 for a mean-zero density rho estimated from n particles:

  ||rho||^2_{H^{-1}}  =  sum_{k != 0}  |rho_hat_k|^2 / lambda_k

where  lambda_k = pi^2 (k1^2 + k2^2)
and    rho_hat_k = (1/n) sum_i  a_{k1} a_{k2} cos(k1 pi x1_i) cos(k2 pi x2_i)
with   a_0 = 1,  a_k = sqrt(2) for k >= 1  (L2 normalisation of Neumann eigenfunctions).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class DivFreeBasis(nn.Module):
    """
    Div-free trigonometric basis on [0,1]^2.

    Parameters
    ----------
    M_max : int
        Maximum wavenumber.  Basis has  L = M_max^2  elements
        with  (m, n) in {1,...,M_max}^2.
    """

    def __init__(self, M_max: int):
        super().__init__()
        self.M_max = M_max

        ms, ns = zip(*[(m, n)
                       for m in range(1, M_max + 1)
                       for n in range(1, M_max + 1)])
        ms = torch.tensor(ms, dtype=torch.float32)   # (L,)
        ns = torch.tensor(ns, dtype=torch.float32)   # (L,)
        # L2 norm of unnormalised e_{mn} on [0,1]^2:
        #   ||e_{mn}||^2 = n^2 pi^2 (1/2)(1/2) + m^2 pi^2 (1/2)(1/2)
        #                = pi^2 (m^2 + n^2) / 4
        # (each sin^2 or cos^2 integrates to 1/2 on [0,1])
        # Normalise to ||e_{mn}||_{L2} = 1:
        norm = 2.0 / (math.pi * torch.sqrt(ms ** 2 + ns ** 2))

        self.register_buffer('ms',   ms)    # (L,)
        self.register_buffer('ns',   ns)    # (L,)
        self.register_buffer('norm', norm)  # (L,)
        self.L = len(ms)

    # ------------------------------------------------------------------
    def eval_basis(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all basis vectors at a set of points.

        Parameters
        ----------
        X : (B, n, 2)  or  (n, 2)

        Returns
        -------
        E : (..., n, L, 2)  –  basis vectors at each point
        """
        squeeze = (X.dim() == 2)
        if squeeze:
            X = X.unsqueeze(0)          # (1, n, 2)

        B, n, _ = X.shape
        x1 = X[..., 0:1]               # (B, n, 1)
        x2 = X[..., 1:2]               # (B, n, 1)

        # broadcast over modes
        m    = self.ms.view(1, 1, -1)  # (1, 1, L)
        nv   = self.ns.view(1, 1, -1)
        norm = self.norm.view(1, 1, -1)
        pi   = math.pi

        e1 =  nv * pi * torch.sin(m  * pi * x1) * torch.cos(nv * pi * x2) * norm  # (B,n,L)
        e2 = -m  * pi * torch.cos(m  * pi * x1) * torch.sin(nv * pi * x2) * norm  # (B,n,L)

        E = torch.stack([e1, e2], dim=-1)  # (B, n, L, 2)
        return E.squeeze(0) if squeeze else E

    # ------------------------------------------------------------------
    def velocity(self, X: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        Compute  v(x) = sum_l  lam_l * e_l(x).

        Parameters
        ----------
        X   : (B, n, 2)
        lam : (B, L)

        Returns
        -------
        v : (B, n, 2)
        """
        E = self.eval_basis(X)                        # (B, n, L, 2)
        return torch.einsum('bl,bnld->bnd', lam, E)   # (B, n, 2)


# ---------------------------------------------------------------------------

class SpectralMixingCost(nn.Module):
    """
    Estimates  ||rho(T)||^2_{H^{-1}([0,1]^2)}  from particle positions.

    Parameters
    ----------
    K_max : int
        Truncation: include modes  (k1, k2) in {0,...,K_max}^2 except (0,0).
    """

    def __init__(self, K_max: int):
        super().__init__()
        self.K_max = K_max

        k1s, k2s = zip(*[
            (k1, k2)
            for k1 in range(0, K_max + 1)
            for k2 in range(0, K_max + 1)
            if not (k1 == 0 and k2 == 0)
        ])
        k1s = torch.tensor(k1s, dtype=torch.float32)
        k2s = torch.tensor(k2s, dtype=torch.float32)

        # Neumann eigenvalues  lambda_k = pi^2 (k1^2 + k2^2)
        eigs = math.pi ** 2 * (k1s ** 2 + k2s ** 2)

        # Normalisation  a_{k} = 1 if k=0,  sqrt(2) if k>=1
        a1 = torch.where(k1s == 0,
                         torch.ones_like(k1s),
                         math.sqrt(2) * torch.ones_like(k1s))
        a2 = torch.where(k2s == 0,
                         torch.ones_like(k2s),
                         math.sqrt(2) * torch.ones_like(k2s))
        ak = a1 * a2   # (K,)

        self.register_buffer('k1s',  k1s)
        self.register_buffer('k2s',  k2s)
        self.register_buffer('eigs', eigs)
        self.register_buffer('ak',   ak)
        self.K = len(k1s)

    # ------------------------------------------------------------------
    def forward(self, X_T: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X_T : (B, n, 2)   terminal particle positions in [0,1]^2

        Returns
        -------
        cost : (B,)   mixing cost  (smaller = better mixed)
        """
        B, n, _ = X_T.shape
        x1 = X_T[..., 0]    # (B, n)
        x2 = X_T[..., 1]    # (B, n)
        pi = math.pi

        k1   = self.k1s.view(1, 1, -1)   # (1, 1, K)
        k2   = self.k2s.view(1, 1, -1)
        ak   = self.ak.view(1, 1, -1)
        eigs = self.eigs.view(1, -1)      # (1, K)

        # Neumann eigenfunction evaluated at each particle
        phi = ak * torch.cos(k1 * pi * x1.unsqueeze(-1)) \
                 * torch.cos(k2 * pi * x2.unsqueeze(-1))   # (B, n, K)

        # Empirical Fourier coefficient  rho_hat_k ≈ (1/n) sum_i phi_k(x_i)
        rho_hat = phi.mean(dim=1)               # (B, K)

        return (rho_hat ** 2 / eigs).sum(-1)    # (B,)


# ---------------------------------------------------------------------------

class DyeMixCost(nn.Module):
    """
    Batched dye-mixing cost:  ||f||^2_{H^{-1}([0,1]^2)}
    where  f_i = +1  for rho_1 particles  and  f_i = -1  for rho_2 particles.

    Measures spatial separation of two labelled particle populations.
    Smaller values = better mixed.

    Parameters
    ----------
    K_max : int
        Spectral truncation: modes (k1,k2) in {0,...,K_max}^2 \\ {(0,0)}.
    """

    def __init__(self, K_max: int):
        super().__init__()
        self.K_max = K_max

        k1s, k2s = zip(*[
            (k1, k2)
            for k1 in range(0, K_max + 1)
            for k2 in range(0, K_max + 1)
            if not (k1 == 0 and k2 == 0)
        ])
        k1s = torch.tensor(k1s, dtype=torch.float32)
        k2s = torch.tensor(k2s, dtype=torch.float32)

        eigs = math.pi ** 2 * (k1s ** 2 + k2s ** 2)

        a1 = torch.where(k1s == 0,
                         torch.ones_like(k1s),
                         math.sqrt(2) * torch.ones_like(k1s))
        a2 = torch.where(k2s == 0,
                         torch.ones_like(k2s),
                         math.sqrt(2) * torch.ones_like(k2s))
        ak = a1 * a2

        self.register_buffer('k1s',  k1s)
        self.register_buffer('k2s',  k2s)
        self.register_buffer('eigs', eigs)
        self.register_buffer('ak',   ak)
        self.K = len(k1s)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor,
                mask1: torch.Tensor | None = None,
                mask2: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        X1    : (B, n1, 2)  rho_1 terminal positions
        X2    : (B, n2, 2)  rho_2 terminal positions
        mask1 : (B, n1)     True for real particles (optional)
        mask2 : (B, n2)     True for real particles (optional)

        Returns
        -------
        cost : (B,)   dye mix-norm  (smaller = better mixed)
        """
        pi = math.pi
        k1 = self.k1s.view(1, 1, -1)   # (1, 1, K)
        k2 = self.k2s.view(1, 1, -1)
        ak = self.ak.view(1, 1, -1)
        eigs = self.eigs.view(1, -1)    # (1, K)

        def _eval_phi(X):
            x1 = X[..., 0]   # (B, n)
            x2 = X[..., 1]
            phi = ak * torch.cos(k1 * pi * x1.unsqueeze(-1)) \
                     * torch.cos(k2 * pi * x2.unsqueeze(-1))  # (B, n, K)
            return phi

        phi1 = _eval_phi(X1)   # (B, n1, K)
        phi2 = _eval_phi(X2)   # (B, n2, K)

        # Weight: +1 for rho_1, -1 for rho_2
        # f_hat_k = (1/N) [ sum_{i in rho_1} phi_k(x_i) - sum_{j in rho_2} phi_k(x_j) ]
        if mask1 is not None:
            phi1 = phi1 * mask1.unsqueeze(-1).float()
        if mask2 is not None:
            phi2 = phi2 * mask2.unsqueeze(-1).float()

        # Count real particles
        if mask1 is not None:
            n1 = mask1.float().sum(-1, keepdim=True)  # (B, 1)
        else:
            n1 = torch.tensor(X1.shape[1], device=X1.device, dtype=X1.dtype)
        if mask2 is not None:
            n2 = mask2.float().sum(-1, keepdim=True)
        else:
            n2 = torch.tensor(X2.shape[1], device=X1.device, dtype=X1.dtype)

        N_total = n1 + n2   # (B, 1) or scalar
        sum1 = phi1.sum(dim=1)    # (B, K)
        sum2 = phi2.sum(dim=1)    # (B, K)
        # Centered signed weights: w_i = +1 (rho_1) / -1 (rho_2),
        # w_tilde_i = w_i - w_bar with w_bar = (n1 - n2)/N, so that the signed
        # empirical measure has exact zero mean (particle analogue of P_0 rho).
        # f_hat_k = (1/N) sum_i w_tilde_i phi_k(X_i)
        #         = (1/N) [ (sum1 - sum2) - w_bar (sum1 + sum2) ]
        w_bar = (n1 - n2) / N_total
        f_hat = ((sum1 - sum2) - w_bar * (sum1 + sum2)) / N_total   # (B, K)

        return (f_hat ** 2 / eigs).sum(-1)   # (B,)
