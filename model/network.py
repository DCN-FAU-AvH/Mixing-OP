"""
model/network.py  --  Two-density transformer for in-context fluid mixing control.

Architecture
------------
The network maps

    ( {x^1_j} ~ rho_1,  {x^2_j} ~ rho_2,  time t )  -->  lambda(t) in R^L

where  v(t, x) = sum_l lambda_l(t) * e_l(x)  is the div-free velocity field.

Two-stage design:

  1. TwoDensityContextEncoder:
       ({x^1_j}, {x^2_j})  -->  z in R^h
     Self-attention on each group, cross-attention between them, then
     joint self-attention and mean-pool.  Called ONCE per trajectory.

  2. CoefficientHead:
       (z, t)  -->  lambda(t)
     Time embedding + MLP.  Called at EVERY ODE step.

The encoder distinguishes rho_1 from rho_2 via learned group embeddings,
enabling the operator to produce different velocity fields for different
spatial configurations of the two densities.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbed(nn.Module):
    """Sinusoidal embedding for a scalar t in [0, 1].  dim must be even."""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(1000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], -1)


# ---------------------------------------------------------------------------
# Cross-attention residual block  (pre-LN)
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Pre-LN cross-attention + FFN residual block.
    Q comes from one group, KV from the other.
    """
    def __init__(self, h: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(h)
        self.norm_kv = nn.LayerNorm(h)
        self.cross_attn = nn.MultiheadAttention(
            h, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, 4 * h),
            nn.GELU(),
            nn.Linear(4 * h, h),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                q_mask: torch.Tensor | None = None,
                kv_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        q  : (B, n_q, h)
        kv : (B, n_kv, h)
        q_mask  : (B, n_q)   True = real token  (optional)
        kv_mask : (B, n_kv)  True = real token  (optional)
        """
        # PyTorch MHA key_padding_mask: True = IGNORE
        kv_pad = (~kv_mask) if kv_mask is not None else None
        q_norm  = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm,
                                       key_padding_mask=kv_pad)
        q = q + attn_out
        q = q + self.ffn(self.norm_ffn(q))
        return q


# ---------------------------------------------------------------------------
# Two-density context encoder
# ---------------------------------------------------------------------------

class TwoDensityContextEncoder(nn.Module):
    """
    Encodes two point clouds {x^1_j} ~ rho_1 and {x^2_j} ~ rho_2
    into a fixed-size vector z in R^h.

    Architecture:
      1. Shared linear lift + learned group embedding
      2. Self-attention on rho_1 tokens  (n_layers_self layers)
      3. Self-attention on rho_2 tokens  (n_layers_self layers)
      4. Cross-attention: rho_1 <-> rho_2  (bidirectional, 1 layer)
      5. Joint self-attention on concatenated tokens  (1 layer)
      6. Masked mean pool -> LayerNorm -> z

    Parameters
    ----------
    d_in          : input spatial dimension (2)
    h             : hidden dimension
    n_heads       : attention heads
    n_layers_self : self-attention depth per group
    n_layers_joint: joint self-attention depth
    dropout       : dropout probability
    """

    def __init__(self, d_in: int, h: int, n_heads: int,
                 n_layers_self: int = 1,
                 n_layers_joint: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.h = h

        # Shared spatial lift
        self.lift = nn.Linear(d_in, h)

        # Learned group embeddings (0 = rho_1, 1 = rho_2)
        self.group_emb = nn.Embedding(2, h)

        # Per-group self-attention
        sa_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=4 * h,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.self_attn_1 = nn.TransformerEncoder(sa_layer, n_layers_self)

        sa_layer_2 = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=4 * h,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.self_attn_2 = nn.TransformerEncoder(sa_layer_2, n_layers_self)

        # Cross-attention (bidirectional)
        self.cross_12 = CrossAttentionBlock(h, n_heads, dropout)
        self.cross_21 = CrossAttentionBlock(h, n_heads, dropout)

        # Joint self-attention on concatenated tokens
        joint_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=4 * h,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.joint_attn = nn.TransformerEncoder(joint_layer, n_layers_joint)

        self.out_norm = nn.LayerNorm(h)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor,
                mask1: torch.Tensor | None = None,
                mask2: torch.Tensor | None = None) -> torch.Tensor:
        """
        X1 : (B, n1, d_in)  rho_1 particles
        X2 : (B, n2, d_in)  rho_2 particles
        mask1 : (B, n1)  True = real particle  (optional)
        mask2 : (B, n2)  True = real particle  (optional)
        returns z : (B, h)
        """
        B = X1.shape[0]
        device = X1.device

        # 1. Shared lift + group embeddings
        h1 = self.lift(X1) + self.group_emb(
            torch.zeros(B, X1.shape[1], dtype=torch.long, device=device))
        h2 = self.lift(X2) + self.group_emb(
            torch.ones(B, X2.shape[1], dtype=torch.long, device=device))

        # Padding masks for PyTorch transformers (True = IGNORE)
        pad1 = (~mask1) if mask1 is not None else None
        pad2 = (~mask2) if mask2 is not None else None

        # 2-3. Per-group self-attention
        h1 = self.self_attn_1(h1, src_key_padding_mask=pad1)
        h2 = self.self_attn_2(h2, src_key_padding_mask=pad2)

        # 4. Cross-attention (bidirectional)
        h1 = self.cross_12(h1, h2, q_mask=mask1, kv_mask=mask2)
        h2 = self.cross_21(h2, h1, q_mask=mask2, kv_mask=mask1)

        # 5. Joint self-attention
        h_all = torch.cat([h1, h2], dim=1)   # (B, n1+n2, h)
        if mask1 is not None and mask2 is not None:
            pad_all = torch.cat([pad1, pad2], dim=1)
        else:
            pad_all = None
        h_all = self.joint_attn(h_all, src_key_padding_mask=pad_all)

        # 6. Masked mean pool
        if mask1 is not None and mask2 is not None:
            real_mask = torch.cat([mask1, mask2], dim=1).float()  # (B, n1+n2)
            h_sum = (h_all * real_mask.unsqueeze(-1)).sum(dim=1)
            n_real = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
            z = h_sum / n_real
        else:
            z = h_all.mean(dim=1)

        return self.out_norm(z)   # (B, h)


# ---------------------------------------------------------------------------
# Coefficient head  (time-conditioned MLP)
# ---------------------------------------------------------------------------

class CoefficientHead(nn.Module):
    """
    Maps (context embedding z, time t) --> basis coefficients lambda(t) in R^L.
    """
    def __init__(self, h: int, L: int, t_emb_dim: int = 64,
                 lam_max: float = 0.5):
        super().__init__()
        self.L = L
        self.lam_max = lam_max
        self.t_embed = SinusoidalEmbed(t_emb_dim)
        self.t_proj  = nn.Linear(t_emb_dim, h)
        self.mlp = nn.Sequential(
            nn.Linear(2 * h, 4 * h),
            nn.GELU(),
            nn.Linear(4 * h, 2 * h),
            nn.GELU(),
            nn.Linear(2 * h, L),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z : (B, h)    precomputed context embedding
        t : (B,)      time values, normalised to [0, 1]
        returns lam : (B, L)
        """
        t_h = self.t_proj(self.t_embed(t))
        raw = self.mlp(torch.cat([z, t_h], dim=-1))
        return self.lam_max * torch.tanh(raw)


# ---------------------------------------------------------------------------
# Full solution-operator network
# ---------------------------------------------------------------------------

class MixingOperatorNet(nn.Module):
    """
    Two-density in-context solution operator for fluid mixing.

    Forward call:
        lambda(t) = net(X1, X2, t)

    where X1 ~ rho_1 and X2 ~ rho_2 encode the two initial densities,
    and lambda(t) are the coefficients of
    v(t,x) = sum_l lambda_l(t) e_l(x).

    Parameters
    ----------
    d             : spatial dimension (2)
    L             : number of div-free basis functions
    h             : transformer hidden dimension
    n_heads       : attention heads
    n_layers_self : self-attention depth per group
    n_layers_joint: joint self-attention depth
    t_emb_dim     : sinusoidal time embedding dimension
    dropout       : dropout probability
    """

    def __init__(
        self,
        d:              int,
        L:              int,
        h:              int   = 256,
        n_heads:        int   = 4,
        n_layers_self:  int   = 1,
        n_layers_joint: int   = 1,
        t_emb_dim:      int   = 64,
        dropout:        float = 0.0,
    ):
        super().__init__()
        self.d, self.L, self.h = d, L, h

        self.ctx_enc = TwoDensityContextEncoder(
            d, h, n_heads, n_layers_self, n_layers_joint, dropout)
        self.coef_head = CoefficientHead(h, L, t_emb_dim)

    def encode_context(self, X1: torch.Tensor, X2: torch.Tensor,
                       mask1: torch.Tensor | None = None,
                       mask2: torch.Tensor | None = None) -> torch.Tensor:
        """
        Precompute context embedding from two density particle sets.
        Call ONCE before the ODE loop.

        X1 : (B, n1, d)   rho_1 particles
        X2 : (B, n2, d)   rho_2 particles
        mask1, mask2 : (B, n1), (B, n2)  True = real particle
        returns z : (B, h)
        """
        return self.ctx_enc(X1, X2, mask1, mask2)

    def forward_from_ctx(self, z: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        """
        Given precomputed context z and current time t, return coefficients.
        Call at EACH time step inside the ODE integrator.
        """
        return self.coef_head(z, t)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor,
                t: torch.Tensor,
                mask1: torch.Tensor | None = None,
                mask2: torch.Tensor | None = None) -> torch.Tensor:
        """
        Convenience: encode context then decode.

        X1 : (B, n1, d)   rho_1 particles
        X2 : (B, n2, d)   rho_2 particles
        t  : (B,)
        returns lam : (B, L)
        """
        z = self.encode_context(X1, X2, mask1, mask2)
        return self.forward_from_ctx(z, t)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
