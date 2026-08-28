"""
Edge-biased multihead attention — v3 centerpiece.

Implements attention where the score between tokens i and j is biased by
a learned function of the current bond-class distribution at edge (i, j):

    attn_score(i, j) = (Q_i · K_j) / sqrt(d_head) + bias_h(bond_ij)

where bias_h is computed from a learnable vector w_h (per head) applied to
the softmaxed bond-class distribution:

    bias_h(bond_ij) = Σ_c softmax(bond_ij / tau)[c] · w_h[c]

Rationale (v3 design spec §2): v2's DiT has attention scores that depend
only on Q/K features; bond predictions are produced independently at the
output layer by the PairwiseBondHead. This means edge predictions can't
coordinate into ring topology, since per-edge CE loss on independent
logits has no mechanism to reward coordinated bond patterns. Edge-biased
attention couples edges through the attention mechanism itself: once a
ring-like aromatic pattern starts to form, ring-seeking heads can route
information along aromatic bonds preferentially, reinforcing the pattern.

Implementation: manual einsum (spec §6.2 Option 1 — readability over speed
for the small T in our problem, max 80 tokens). Uses standard PyTorch
operations; no Flash Attention needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBiasedMultiheadAttention(nn.Module):
    """
    Multihead attention with additive bond-class bias per edge.

    Forward signature matches nn.MultiheadAttention except for:
      - `edge_probs` : (B, T_q, T_k, n_bond_classes) soft bond distribution
                       between query and key positions. Required when
                       bias_enabled=True; ignored otherwise.
      - `key_padding_mask` : (B, T_k) bool, True at positions to IGNORE
                              (matches PyTorch's convention).

    Compatible with self-attention (T_q == T_k) and cross-attention
    (T_q ≠ T_k). In cross-attention, `edge_probs[b, i, j]` describes the
    bond state between query i and key j — for Stage 2 this corresponds
    to terminal-to-scaffold bonds.

    Parameters
    ----------
    embed_dim      : total d_model
    num_heads      : number of attention heads
    dropout        : dropout probability on attention weights
    bias_enabled   : whether to build and apply bond-class bias (False
                      reproduces standard multihead attention exactly,
                      useful for the EDGE_ATTN_ENABLED=False ablation)
    n_bond_classes : 5 for our scheme (none, single, aromatic, double, triple)
    bias_temperature : tau for softmax over bond_probs before bias projection.
                        Lower = sharper commitment to current bond state.
                        Default 1.0 (no sharpening) per spec §9 Q1 default.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0,
                 bias_enabled=True, n_bond_classes=5,
                 bias_temperature=1.0):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.dropout = dropout
        self.bias_enabled = bias_enabled
        self.n_bond_classes = n_bond_classes
        self.bias_temperature = bias_temperature

        # Standard Q, K, V projections (combined for efficiency in self-attn;
        # kept separate here to support cross-attention with different input
        # sources cleanly).
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if bias_enabled:
            # Learnable bond-class weights, one vector per attention head.
            # Init to zero so initial behavior matches standard attention
            # (additive bias of 0) and the model has to actively learn
            # whether each head benefits from edge biasing.
            self.bond_bias = nn.Parameter(
                torch.zeros(num_heads, n_bond_classes)
            )
        else:
            self.register_parameter("bond_bias", None)

    def forward(self, query, key, value, edge_probs=None,
                key_padding_mask=None):
        """
        Parameters
        ----------
        query, key, value : (B, T_q/T_k/T_k, embed_dim)
        edge_probs        : (B, T_q, T_k, n_bond_classes) — soft bond probs
                             between each (query, key) position pair.
                             Can be None when bias_enabled=False.
        key_padding_mask  : (B, T_k) bool, True at positions to mask out.

        Returns
        -------
        out : (B, T_q, embed_dim)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        H = self.num_heads
        D = self.d_head

        # Project to Q, K, V and reshape to (B, H, T, D)
        Q = self.q_proj(query).reshape(B, T_q, H, D).transpose(1, 2)
        K = self.k_proj(key).reshape(B, T_k, H, D).transpose(1, 2)
        V = self.v_proj(value).reshape(B, T_k, H, D).transpose(1, 2)

        # Attention scores: (B, H, T_q, T_k)
        scale = 1.0 / math.sqrt(D)
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale

        # ── Edge bias ────────────────────────────────────────────────────
        # Spec §2.4: bias_h(edge_ij) = Σ_c softmax(bond_ij/tau)[c] · w_h[c]
        # Each head learns its own weighting of bond classes.
        if self.bias_enabled:
            if edge_probs is None:
                raise ValueError(
                    "bias_enabled=True but edge_probs is None. Pass bond "
                    "probabilities between query/key positions."
                )
            # edge_probs is the RAW xt_bond (one-hot-with-noise), so apply
            # softmax with temperature to get well-defined probabilities.
            bond_probs = F.softmax(
                edge_probs / self.bias_temperature, dim=-1
            )                                       # (B, T_q, T_k, C)

            # Project to scalar per head:
            # bias[b, h, q, k] = Σ_c bond_probs[b, q, k, c] * bond_bias[h, c]
            bias = torch.einsum(
                "bqkc,hc->bhqk", bond_probs, self.bond_bias
            )                                       # (B, H, T_q, T_k)
            scores = scores + bias

        # ── Padding mask ────────────────────────────────────────────────
        # Mask out invalid KEY positions (padding atoms should not be
        # attended to). Shape matches scores via broadcast.
        if key_padding_mask is not None:
            # True in mask = ignore, so set logits to -inf before softmax
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1),
                float("-inf"),
            )

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        # Guard against NaN when a query has no valid keys (whole row -inf).
        # Replace any NaN in attention weights with 0; the output for that
        # query position will just be zero (which is fine; it's a padding
        # position that will be masked at the loss level).
        attn = torch.nan_to_num(attn, nan=0.0)

        # Aggregate: (B, H, T_q, D)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, V)

        # Recombine heads: (B, T_q, embed_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, T_q, self.embed_dim)
        out = self.out_proj(out)

        return out
