"""
HiMoFlow v5.4 — Ring Layout Diffusion (A1, Batch 3).
=====================================================

Discrete absorbing-state diffusion over the layout state
(R, F, L, P_len, P_pos). A1's job: predict a clean layout given
(solubility, gap) conditioning. A2 (Batch 4) will then assign atom
IDs given the layout. The deterministic decoder
(ring_layout_decoder.decode_layout_to_scaffold) composes layout +
atom_ids into v5.3-compatible scaffold tensors for Stage 2.

Architecture
------------
Token sequence of 26 tokens, in this fixed order:

   tokens[0..3]    — ring tokens  (one per ring slot k ∈ [0, R_MAX))
                     each predicts R[k] over RING_PAD..RING_5_ALIPH (5 cls)

   tokens[4..9]    — pair tokens  (one per upper-tri (i,j) with i<j)
                     ordered (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
                     each predicts F[i,j] (3 cls) AND L[i,j] (5 cls)

   tokens[10..25]  — pendant tokens  (one per (ring_i, slot_p))
                     ordered (0,0), (0,1), ..., (3,3)
                     each predicts P_len[i,p] (6 cls) AND P_pos[i,p] (6 cls)

Each token's input embedding combines a learnable role+slot embedding
with one or two value embeddings (one per categorical variable carried
by the token). Value embeddings include a MASK class on the input side
only — output heads predict over the original (no-MASK) vocab.

Diffusion is absorbing-state with a cosine schedule on α(t).
   At training: each variable's element is replaced with MASK
                independently with probability α(t).
   At sampling: x_T = all-MASK; iteratively unmask the most-confident
                positions until x_0 is reached.

Loss: cross-entropy on every position (no masked-only filter), matching
v5.3's terminal_fragment_diffusion convention. The layout state has no
"padding" notion — every position must be predicted. PAD is just one
of R's classes.

Sample-time post-processing
---------------------------
The raw token-wise prediction can produce layouts that violate the
decoder's structural constraints. We apply minimal post-processing
before returning:

  (1) Mirror F upper triangle to lower (symmetric).
  (2) Zero L where F!=2 (linker only meaningful when linked).
  (3) Zero P_pos where P_len==0 (position only meaningful with pendant).
  (4) Left-pack R: if R[k]=PAD with R[k+1]!=PAD, slide non-PAD entries
      to the front.

After post-processing, decode_layout_to_scaffold validates the layout
and rejects samples that still violate constraints (e.g., disconnected
ring graph, multi-anchor topology, pendant on a fusion edge). The
rejection rate is the primary B3 success metric: a healthy A1 should
produce decodable layouts most of the time.

v5.4-1 caveat: the pendant heads will collapse to predicting class 0
everywhere because RedDB's stripped scaffolds have zero pendants. This
is correct behavior for RedDB but means transferring to a pendant-rich
dataset (ZINC250K, QM9) requires a full retrain — pendant weights
carry no useful prior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Vocab constants (mirror ring_layout_decoder.py) ───────────────────

R_MAX = 4         # max rings per molecule
P_MAX = 4         # max pendant slots per ring
L_MAX = 4         # max linker length
P_LEN_MAX = 5     # max pendant chain length

# Class counts for each variable (excluding MASK)
N_R_CLASSES = 5         # PAD, 6-arom, 6-aliph, 5-arom, 5-aliph
N_F_CLASSES = 3         # NONE, FUSED, LINKED
N_L_CLASSES = L_MAX + 1     # 0..4 → 5 classes
N_PLEN_CLASSES = P_LEN_MAX + 1  # 0..5 → 6 classes
N_PPOS_CLASSES = 6      # 0..5 (covers ring sizes 5 and 6)

# MASK class IDs (one beyond the original vocab for each variable)
MASK_R = N_R_CLASSES
MASK_F = N_F_CLASSES
MASK_L = N_L_CLASSES
MASK_PLEN = N_PLEN_CLASSES
MASK_PPOS = N_PPOS_CLASSES

# Token-sequence layout
N_RING_TOKENS = R_MAX                                # 4
N_PAIR_TOKENS = R_MAX * (R_MAX - 1) // 2             # 6 (upper triangle)
N_PEND_TOKENS = R_MAX * P_MAX                        # 16
N_TOKENS = N_RING_TOKENS + N_PAIR_TOKENS + N_PEND_TOKENS  # 26

# Token slice indices
RING_TOKEN_START = 0
RING_TOKEN_END = N_RING_TOKENS                       # 4
PAIR_TOKEN_START = RING_TOKEN_END                    # 4
PAIR_TOKEN_END = PAIR_TOKEN_START + N_PAIR_TOKENS    # 10
PEND_TOKEN_START = PAIR_TOKEN_END                    # 10
PEND_TOKEN_END = PEND_TOKEN_START + N_PEND_TOKENS    # 26


def _upper_tri_pairs(r_max: int = R_MAX) -> List[Tuple[int, int]]:
    """Returns the canonical list of upper-triangular (i,j) pairs with
    i < j, ordered first by i then j. For R_MAX=4:
    [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]."""
    return [(i, j) for i in range(r_max) for j in range(i + 1, r_max)]


PAIR_INDICES: List[Tuple[int, int]] = _upper_tri_pairs()
assert len(PAIR_INDICES) == N_PAIR_TOKENS


def _pair_index(i: int, j: int) -> int:
    """Flat index for upper-triangular pair (i,j) with i<j."""
    if i >= j:
        i, j = j, i
    # Sum over rows 0..i-1 plus offset within row i
    return i * R_MAX - (i * (i + 1)) // 2 + (j - i - 1)


# Sanity: verify ordering
for _idx, (_i, _j) in enumerate(PAIR_INDICES):
    assert _pair_index(_i, _j) == _idx, "pair index mismatch"


# ─── Diffusion noise schedule ──────────────────────────────────────────

def alpha_bar(t: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    """Mask probability α(t) ∈ [0, 1]. α(0)=0, α(1)=1.

    Cosine: α(t) = 1 - cos²(πt/2). Concave: more masking late.
    Linear: α(t) = t.
    """
    if schedule == "cosine":
        return 1.0 - torch.cos(math.pi * t / 2.0) ** 2
    elif schedule == "linear":
        return t.clone()
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def corrupt_categorical(
    x_0: torch.Tensor,
    alpha: torch.Tensor,
    mask_class_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace each element of `x_0` with `mask_class_id` independently
    with probability `alpha`. Returns (x_t, is_masked).

    x_0: (B, ...) int
    alpha: (B,) float — probability per batch element
    """
    B = x_0.shape[0]
    # Reshape alpha for broadcasting
    extra_dims = (1,) * (x_0.dim() - 1)
    alpha_b = alpha.view(B, *extra_dims)
    rand = torch.rand_like(x_0, dtype=torch.float32)
    is_masked = rand < alpha_b
    x_t = torch.where(is_masked, torch.full_like(x_0, mask_class_id), x_0)
    return x_t, is_masked


# ─── Model components ──────────────────────────────────────────────────

class SinusoidalTimeEmbed(nn.Module):
    """Standard sinusoidal embedding for diffusion timestep."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaLN(nn.Module):
    """Adaptive LayerNorm: γ, β predicted from a conditioning vector.

    Output = LN(x) * (1 + γ) + β where (γ, β) = Linear(SiLU(cond)).
    Following DiT's design, we initialize γ, β to zero so the block
    is identity at start.
    """
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 2 * d_model),
        )
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d), cond: (B, d_cond)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class DiTBlock(nn.Module):
    """Standard DiT block: AdaLN → MHA → AdaLN → FFN, both with residuals.

    No edge-bias on attention: A1's attention is over an abstract 26-token
    sequence whose pairwise structure is itself the prediction target,
    not a fixed input graph to bias on. Pure self-attention is correct.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_cond: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adaln1 = AdaLN(d_model, d_cond)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.adaln2 = AdaLN(d_model, d_cond)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)
        h = self.adaln2(x, cond)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


# ─── Main model ────────────────────────────────────────────────────────

class RingLayoutDiffusion(nn.Module):
    """Discrete absorbing-state diffusion over the layout state.

    Parameters
    ----------
    d_model, n_layers, n_heads, d_ff : DiT capacity knobs.
    d_cond : conditioning dimension (used internally; cond + time go
             through a small MLP into this dim).
    condition_dim : external conditioning vector size (default 2 for
                    (solubility, gap)).
    cfg_drop_prob : at training, probability of dropping the condition
                    (replacing with zeros) for classifier-free guidance.
    schedule : "cosine" or "linear".
    """
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        d_cond: Optional[int] = None,
        condition_dim: int = 2,
        cfg_drop_prob: float = 0.1,
        dropout: float = 0.1,
        schedule: str = "cosine",
        time_embed_dim: int = 64,
        r_max: int = R_MAX,
        p_max: int = P_MAX,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        if d_cond is None:
            d_cond = d_model

        if r_max != R_MAX or p_max != P_MAX:
            raise ValueError(
                f"r_max/p_max must match decoder constants ({R_MAX}, {P_MAX}); "
                f"got ({r_max}, {p_max}). Architecturally A1 supports other "
                f"sizes but the decoder side does not yet."
            )

        self.d_model = d_model
        self.d_cond = d_cond
        self.condition_dim = condition_dim
        self.cfg_drop_prob = cfg_drop_prob
        self.schedule = schedule
        self.r_max = r_max
        self.p_max = p_max
        self.n_tokens = N_TOKENS

        # ── Per-token role+slot positional embedding ────────────────
        # 26 unique learnable vectors, one per token slot.
        self.token_pos_embed = nn.Embedding(N_TOKENS, d_model)

        # ── Value embeddings (input side has +1 MASK class) ─────────
        self.r_value_embed = nn.Embedding(N_R_CLASSES + 1, d_model)
        self.f_value_embed = nn.Embedding(N_F_CLASSES + 1, d_model)
        self.l_value_embed = nn.Embedding(N_L_CLASSES + 1, d_model)
        self.plen_value_embed = nn.Embedding(N_PLEN_CLASSES + 1, d_model)
        self.ppos_value_embed = nn.Embedding(N_PPOS_CLASSES + 1, d_model)

        # ── Time + condition embedding ──────────────────────────────
        self.time_embed = SinusoidalTimeEmbed(time_embed_dim)
        self.cond_in_proj = nn.Sequential(
            nn.Linear(condition_dim + time_embed_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # ── DiT stack ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                d_cond=d_cond,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Output heads (each predicts over no-MASK vocab) ─────────
        # Per-role: ring (1 head), pair (2 heads), pendant (2 heads).
        # NOTE: heads use default PyTorch init (Kaiming-uniform). Zero-
        # init was tried but kills gradient through the DiT stack on
        # the first step (∂L/∂h = grad_out @ W^T = 0 with W=0). We rely
        # on AdaLN zero-init instead to make blocks identity-on-input.
        self.r_head = nn.Linear(d_model, N_R_CLASSES)
        self.f_head = nn.Linear(d_model, N_F_CLASSES)
        self.l_head = nn.Linear(d_model, N_L_CLASSES)
        self.plen_head = nn.Linear(d_model, N_PLEN_CLASSES)
        self.ppos_head = nn.Linear(d_model, N_PPOS_CLASSES)

        # Zero-init the bias only — gives a softer "near-uniform" prior
        # at start without breaking gradient flow.
        for head in (self.r_head, self.f_head, self.l_head,
                     self.plen_head, self.ppos_head):
            nn.init.zeros_(head.bias)

        # Cache pair indices as buffer for index gather/scatter
        pair_i = torch.tensor([i for i, _ in PAIR_INDICES], dtype=torch.long)
        pair_j = torch.tensor([j for _, j in PAIR_INDICES], dtype=torch.long)
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)

    # ──────────────────────────────────────────────────────────────────
    #  Helpers — pack / unpack the layout to/from token-aligned tensors
    # ──────────────────────────────────────────────────────────────────

    def gather_pair_upper(self, mat_BNN: torch.Tensor) -> torch.Tensor:
        """Extract upper-triangular entries of (B, R_MAX, R_MAX) into
        (B, N_PAIR_TOKENS) following the canonical PAIR_INDICES order."""
        return mat_BNN[:, self.pair_i, self.pair_j]

    def scatter_pair_upper_symmetric(
        self, vec_BP: torch.Tensor
    ) -> torch.Tensor:
        """Inverse of gather_pair_upper: build a symmetric (B, R_MAX, R_MAX)
        matrix with the supplied upper-triangle, mirrored to lower, with
        zero diagonal."""
        B = vec_BP.shape[0]
        out = torch.zeros(
            B, self.r_max, self.r_max,
            dtype=vec_BP.dtype, device=vec_BP.device,
        )
        out[:, self.pair_i, self.pair_j] = vec_BP
        out[:, self.pair_j, self.pair_i] = vec_BP
        return out

    def gather_pendant_flat(self, mat_BRP: torch.Tensor) -> torch.Tensor:
        """Flatten (B, R_MAX, P_MAX) → (B, N_PEND_TOKENS) row-major."""
        return mat_BRP.reshape(mat_BRP.shape[0], -1)

    def scatter_pendant_flat(self, vec_BS: torch.Tensor) -> torch.Tensor:
        """Inverse of gather_pendant_flat."""
        return vec_BS.reshape(vec_BS.shape[0], self.r_max, self.p_max)

    # ──────────────────────────────────────────────────────────────────
    #  Token assembly
    # ──────────────────────────────────────────────────────────────────

    def _assemble_input_tokens(
        self,
        R_t: torch.Tensor,         # (B, R_MAX)
        F_upper_t: torch.Tensor,   # (B, N_PAIR_TOKENS)
        L_upper_t: torch.Tensor,   # (B, N_PAIR_TOKENS)
        Plen_flat_t: torch.Tensor, # (B, N_PEND_TOKENS)
        Ppos_flat_t: torch.Tensor, # (B, N_PEND_TOKENS)
    ) -> torch.Tensor:
        """Build the (B, N_TOKENS, d_model) input tensor by combining
        position embeddings with value embeddings.

        Ring tokens get pos + r_value.
        Pair tokens get pos + f_value + l_value.
        Pendant tokens get pos + plen_value + ppos_value.
        """
        B = R_t.shape[0]
        device = R_t.device

        # All position embeddings at once
        all_positions = torch.arange(N_TOKENS, device=device)
        pos_emb = self.token_pos_embed(all_positions)  # (N_TOKENS, d_model)
        # Expand to batch
        tokens = pos_emb.unsqueeze(0).expand(B, -1, -1).clone()  # (B, N_TOKENS, d)

        # Ring token slots [0..3]
        r_emb = self.r_value_embed(R_t)  # (B, R_MAX, d)
        tokens[:, RING_TOKEN_START:RING_TOKEN_END] = (
            tokens[:, RING_TOKEN_START:RING_TOKEN_END] + r_emb
        )

        # Pair token slots [4..9]
        f_emb = self.f_value_embed(F_upper_t)  # (B, N_PAIR_TOKENS, d)
        l_emb = self.l_value_embed(L_upper_t)  # (B, N_PAIR_TOKENS, d)
        tokens[:, PAIR_TOKEN_START:PAIR_TOKEN_END] = (
            tokens[:, PAIR_TOKEN_START:PAIR_TOKEN_END] + f_emb + l_emb
        )

        # Pendant token slots [10..25]
        plen_emb = self.plen_value_embed(Plen_flat_t)
        ppos_emb = self.ppos_value_embed(Ppos_flat_t)
        tokens[:, PEND_TOKEN_START:PEND_TOKEN_END] = (
            tokens[:, PEND_TOKEN_START:PEND_TOKEN_END] + plen_emb + ppos_emb
        )

        return tokens

    def _build_cond(
        self,
        condition: torch.Tensor,     # (B, condition_dim)
        alpha: torch.Tensor,         # (B,) in [0, 1]
    ) -> torch.Tensor:
        """Produce a (B, d_cond) conditioning vector from the external
        condition + diffusion timestep alpha."""
        t_emb = self.time_embed(alpha)            # (B, time_embed_dim)
        cat = torch.cat([condition, t_emb], dim=-1)
        return self.cond_in_proj(cat)             # (B, d_cond)

    # ──────────────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        R_t: torch.Tensor,           # (B, R_MAX) noisy R values, in [0, N_R_CLASSES] (incl. MASK)
        F_full_t: torch.Tensor,      # (B, R_MAX, R_MAX) noisy F (will be reduced to upper)
        L_full_t: torch.Tensor,      # (B, R_MAX, R_MAX) noisy L (will be reduced to upper)
        Plen_t: torch.Tensor,        # (B, R_MAX, P_MAX) noisy P_len
        Ppos_t: torch.Tensor,        # (B, R_MAX, P_MAX) noisy P_pos
        alpha: torch.Tensor,         # (B,) in [0, 1]
        condition: torch.Tensor,     # (B, condition_dim)
    ) -> Dict[str, torch.Tensor]:
        """Returns a dict of logits:
            R_logits     : (B, R_MAX, N_R_CLASSES)
            F_logits     : (B, N_PAIR_TOKENS, N_F_CLASSES)
            L_logits     : (B, N_PAIR_TOKENS, N_L_CLASSES)
            Plen_logits  : (B, R_MAX, P_MAX, N_PLEN_CLASSES)
            Ppos_logits  : (B, R_MAX, P_MAX, N_PPOS_CLASSES)
        """
        B = R_t.shape[0]

        # Reduce F, L to upper-triangular slices
        F_upper = self.gather_pair_upper(F_full_t)
        L_upper = self.gather_pair_upper(L_full_t)

        # Flatten pendant tensors
        Plen_flat = self.gather_pendant_flat(Plen_t)
        Ppos_flat = self.gather_pendant_flat(Ppos_t)

        # CFG: at training, drop condition with prob cfg_drop_prob
        if self.training and self.cfg_drop_prob > 0:
            drop = (torch.rand(B, device=condition.device)
                    < self.cfg_drop_prob).unsqueeze(-1)
            condition = torch.where(drop, torch.zeros_like(condition),
                                    condition)

        # Build token input
        tokens = self._assemble_input_tokens(
            R_t, F_upper, L_upper, Plen_flat, Ppos_flat
        )  # (B, N_TOKENS, d_model)

        # Conditioning context
        cond_emb = self._build_cond(condition, alpha)  # (B, d_cond)

        # Run DiT blocks
        h = tokens
        for blk in self.blocks:
            h = blk(h, cond_emb)
        h = self.final_norm(h)

        # Slice tokens by role
        h_ring = h[:, RING_TOKEN_START:RING_TOKEN_END]   # (B, R_MAX, d)
        h_pair = h[:, PAIR_TOKEN_START:PAIR_TOKEN_END]   # (B, N_PAIR_TOKENS, d)
        h_pend = h[:, PEND_TOKEN_START:PEND_TOKEN_END]   # (B, N_PEND_TOKENS, d)

        # Apply heads
        R_logits = self.r_head(h_ring)                   # (B, R_MAX, N_R)
        F_logits = self.f_head(h_pair)                   # (B, N_PAIR, N_F)
        L_logits = self.l_head(h_pair)                   # (B, N_PAIR, N_L)
        Plen_h = h_pend.reshape(B, self.r_max, self.p_max, -1)
        Plen_logits = self.plen_head(Plen_h)              # (B, R_MAX, P_MAX, N_PLEN)
        Ppos_logits = self.ppos_head(Plen_h)              # (B, R_MAX, P_MAX, N_PPOS)

        return {
            "R_logits": R_logits,
            "F_logits": F_logits,
            "L_logits": L_logits,
            "Plen_logits": Plen_logits,
            "Ppos_logits": Ppos_logits,
        }

    # ──────────────────────────────────────────────────────────────────
    #  Training: corrupt + forward + CE
    # ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        R: torch.Tensor,             # (B, R_MAX) clean
        F_mat: torch.Tensor,         # (B, R_MAX, R_MAX) clean
        L_mat: torch.Tensor,         # (B, R_MAX, R_MAX) clean
        P_len: torch.Tensor,         # (B, R_MAX, P_MAX) clean
        P_pos: torch.Tensor,         # (B, R_MAX, P_MAX) clean
        condition: torch.Tensor,     # (B, condition_dim)
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """One training step. Sample timestep, corrupt, predict, CE.

        Loss is computed on EVERY position (no masked-only filter),
        matching v5.3's terminal_fragment_diffusion convention. The
        layout state has no padding; PAD is just one of R's classes.

        Parameter names use _mat suffix on F/L matrices to avoid name
        collision with torch.nn.functional (imported as F by convention).
        """
        device = R.device
        B = R.shape[0]

        # Sample diffusion timestep
        t = torch.empty(B, device=device).uniform_(alpha_min, alpha_max)
        alpha = alpha_bar(t, self.schedule)

        # Independent corruption per variable
        R_t, _ = corrupt_categorical(R, alpha, MASK_R)
        F_t, _ = corrupt_categorical(F_mat, alpha, MASK_F)
        L_t, _ = corrupt_categorical(L_mat, alpha, MASK_L)
        Plen_t, _ = corrupt_categorical(P_len, alpha, MASK_PLEN)
        Ppos_t, _ = corrupt_categorical(P_pos, alpha, MASK_PPOS)

        out = self.forward(
            R_t=R_t,
            F_full_t=F_t,
            L_full_t=L_t,
            Plen_t=Plen_t,
            Ppos_t=Ppos_t,
            alpha=alpha,
            condition=condition,
        )

        # Targets for upper-tri pair losses
        F_upper_target = self.gather_pair_upper(F_mat)   # (B, N_PAIR_TOKENS)
        L_upper_target = self.gather_pair_upper(L_mat)   # (B, N_PAIR_TOKENS)

        # CE per head (mean over all positions)
        loss_R = F.cross_entropy(
            out["R_logits"].reshape(-1, N_R_CLASSES),
            R.reshape(-1),
        )
        loss_F = F.cross_entropy(
            out["F_logits"].reshape(-1, N_F_CLASSES),
            F_upper_target.reshape(-1),
        )
        loss_L = F.cross_entropy(
            out["L_logits"].reshape(-1, N_L_CLASSES),
            L_upper_target.reshape(-1),
        )
        loss_Plen = F.cross_entropy(
            out["Plen_logits"].reshape(-1, N_PLEN_CLASSES),
            P_len.reshape(-1),
        )
        loss_Ppos = F.cross_entropy(
            out["Ppos_logits"].reshape(-1, N_PPOS_CLASSES),
            P_pos.reshape(-1),
        )

        # Per-head accuracies (with no_grad)
        with torch.no_grad():
            acc_R = (out["R_logits"].argmax(-1) == R).float().mean()
            acc_F = (out["F_logits"].argmax(-1) == F_upper_target).float().mean()
            acc_L = (out["L_logits"].argmax(-1) == L_upper_target).float().mean()
            acc_Plen = (out["Plen_logits"].argmax(-1) == P_len).float().mean()
            acc_Ppos = (out["Ppos_logits"].argmax(-1) == P_pos).float().mean()

        # Combine
        if loss_weights is None:
            loss_weights = {"R": 1.0, "F": 1.0, "L": 1.0,
                            "Plen": 1.0, "Ppos": 1.0}
        loss = (
            loss_weights["R"] * loss_R
            + loss_weights["F"] * loss_F
            + loss_weights["L"] * loss_L
            + loss_weights["Plen"] * loss_Plen
            + loss_weights["Ppos"] * loss_Ppos
        )

        return {
            "loss": loss,
            "loss_R": loss_R.detach(),
            "loss_F": loss_F.detach(),
            "loss_L": loss_L.detach(),
            "loss_Plen": loss_Plen.detach(),
            "loss_Ppos": loss_Ppos.detach(),
            "acc_R": acc_R,
            "acc_F": acc_F,
            "acc_L": acc_L,
            "acc_Plen": acc_Plen,
            "acc_Ppos": acc_Ppos,
        }

    # ──────────────────────────────────────────────────────────────────
    #  Sampling: iterative absorbing-state unmasking
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,     # (B, condition_dim)
        n_steps: int = 20,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        seed: Optional[int] = None,
        post_process: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Iterative confidence-based unmasking sampler.

        Schedule: at step s ∈ {0..n_steps-1}, the target masked
        fraction at the END of this step is α((n_steps-s-1)/n_steps).
        We compute how many positions to unmask this step and pick the
        most-confident ones globally (across all heads, weighted by
        the predicted prob of the argmax class).

        cfg_scale > 1.0 enables classifier-free guidance: logits are
        guided as cond_logits + cfg_scale*(cond_logits - uncond_logits).

        Returns a dict with the post-processed clean layout state.
        """
        device = condition.device
        B = condition.shape[0]
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Initialize: all MASK
        R_t = torch.full((B, self.r_max), MASK_R, dtype=torch.long, device=device)
        F_t = torch.full((B, self.r_max, self.r_max), MASK_F,
                         dtype=torch.long, device=device)
        # F diagonal must always be 0 (no self-relation); zero those out
        # and treat as "fixed/unmaskable".
        diag_idx = torch.arange(self.r_max, device=device)
        F_t[:, diag_idx, diag_idx] = 0
        L_t = torch.full((B, self.r_max, self.r_max), MASK_L,
                         dtype=torch.long, device=device)
        L_t[:, diag_idx, diag_idx] = 0
        Plen_t = torch.full((B, self.r_max, self.p_max), MASK_PLEN,
                            dtype=torch.long, device=device)
        Ppos_t = torch.full((B, self.r_max, self.p_max), MASK_PPOS,
                            dtype=torch.long, device=device)

        # Track maskedness flags per element
        R_masked = torch.ones_like(R_t, dtype=torch.bool)
        F_masked = torch.ones_like(F_t, dtype=torch.bool)
        F_masked[:, diag_idx, diag_idx] = False  # diagonal never masked
        # F symmetry: only sample upper triangle, mirror below.
        # Make lower triangle "not currently masked" (we'll reuse upper
        # values to fill it after each step).
        for i in range(self.r_max):
            for j in range(i):
                F_masked[:, i, j] = False
        L_masked = torch.ones_like(L_t, dtype=torch.bool)
        L_masked[:, diag_idx, diag_idx] = False
        for i in range(self.r_max):
            for j in range(i):
                L_masked[:, i, j] = False
        Plen_masked = torch.ones_like(Plen_t, dtype=torch.bool)
        Ppos_masked = torch.ones_like(Ppos_t, dtype=torch.bool)

        # Total maskable positions at start
        total_masked_start = (
            R_masked.sum().item()
            + F_masked.sum().item()
            + L_masked.sum().item()
            + Plen_masked.sum().item()
            + Ppos_masked.sum().item()
        )

        for step in range(n_steps):
            # Target masked fraction at the END of this step
            t_next = torch.tensor(
                [(n_steps - step - 1) / n_steps],
                device=device, dtype=torch.float32,
            )
            alpha_next = alpha_bar(t_next, self.schedule).item()
            t_now = torch.tensor(
                [(n_steps - step) / n_steps],
                device=device, dtype=torch.float32,
            ).expand(B)

            # Mirror F upper → lower for the forward pass
            F_t_full = self._mirror_upper_to_full(F_t)
            L_t_full = self._mirror_upper_to_full(L_t)

            out = self.forward(
                R_t=R_t,
                F_full_t=F_t_full,
                L_full_t=L_t_full,
                Plen_t=Plen_t,
                Ppos_t=Ppos_t,
                alpha=t_now,
                condition=condition,
            )

            if cfg_scale != 1.0:
                # Classifier-free guidance: also forward with zero condition
                uncond = torch.zeros_like(condition)
                out_uncond = self.forward(
                    R_t=R_t,
                    F_full_t=F_t_full,
                    L_full_t=L_t_full,
                    Plen_t=Plen_t,
                    Ppos_t=Ppos_t,
                    alpha=t_now,
                    condition=uncond,
                )
                for k in out:
                    out[k] = out_uncond[k] + cfg_scale * (out[k] - out_uncond[k])

            # Apply temperature
            for k in out:
                out[k] = out[k] / max(temperature, 1e-6)

            # Compute per-element confidence (max softmax prob) and predicted class
            R_probs = F.softmax(out["R_logits"], dim=-1)
            R_conf, R_pred = R_probs.max(dim=-1)         # (B, R_MAX)
            F_probs = F.softmax(out["F_logits"], dim=-1)
            F_conf_upper, F_pred_upper = F_probs.max(dim=-1)  # (B, N_PAIR)
            L_probs = F.softmax(out["L_logits"], dim=-1)
            L_conf_upper, L_pred_upper = L_probs.max(dim=-1)
            Plen_probs = F.softmax(out["Plen_logits"], dim=-1)
            Plen_conf, Plen_pred = Plen_probs.max(dim=-1)
            Ppos_probs = F.softmax(out["Ppos_logits"], dim=-1)
            Ppos_conf, Ppos_pred = Ppos_probs.max(dim=-1)

            # Scatter pair predictions to full (B, R_MAX, R_MAX)
            F_conf_full = torch.zeros_like(F_t, dtype=torch.float32)
            F_conf_full[:, self.pair_i, self.pair_j] = F_conf_upper
            F_pred_full = F_t.clone()
            F_pred_full[:, self.pair_i, self.pair_j] = F_pred_upper
            L_conf_full = torch.zeros_like(L_t, dtype=torch.float32)
            L_conf_full[:, self.pair_i, self.pair_j] = L_conf_upper
            L_pred_full = L_t.clone()
            L_pred_full[:, self.pair_i, self.pair_j] = L_pred_upper

            # Compute number of MASKs to remove this step
            # Total maskable elements still needing to be unmasked:
            n_masked_R = R_masked.sum().item()
            n_masked_F = F_masked.sum().item()
            n_masked_L = L_masked.sum().item()
            n_masked_Plen = Plen_masked.sum().item()
            n_masked_Ppos = Ppos_masked.sum().item()
            n_currently = (n_masked_R + n_masked_F + n_masked_L
                           + n_masked_Plen + n_masked_Ppos)

            n_target = int(round(total_masked_start * alpha_next))
            n_to_unmask = max(n_currently - n_target, 0)
            if step == n_steps - 1:
                n_to_unmask = n_currently  # final step: unmask everything

            if n_to_unmask <= 0:
                continue

            # Build a global confidence vector over all currently-masked
            # positions, then pick top-K.
            # Each position gets:
            #   confidence_score = predicted prob
            # We sample with confidence (deterministic argmax), but to
            # add diversity we add a small Gumbel perturbation.
            def gumbel_perturb(conf):
                u = torch.rand_like(conf)
                u = u.clamp(min=1e-9, max=1.0 - 1e-9)
                return conf - torch.log(-torch.log(u))

            # Each maskable position contributes one score; non-masked
            # positions get -inf so they're never chosen.
            scores_R = torch.where(
                R_masked, gumbel_perturb(R_conf),
                torch.full_like(R_conf, float("-inf")),
            )
            scores_F = torch.where(
                F_masked, gumbel_perturb(F_conf_full),
                torch.full_like(F_conf_full, float("-inf")),
            )
            scores_L = torch.where(
                L_masked, gumbel_perturb(L_conf_full),
                torch.full_like(L_conf_full, float("-inf")),
            )
            scores_Plen = torch.where(
                Plen_masked, gumbel_perturb(Plen_conf),
                torch.full_like(Plen_conf, float("-inf")),
            )
            scores_Ppos = torch.where(
                Ppos_masked, gumbel_perturb(Ppos_conf),
                torch.full_like(Ppos_conf, float("-inf")),
            )

            # Flatten all scores into one global vector and pick top n_to_unmask
            # Per-batch element: keep them grouped so sampling is
            # consistent across rows. Easiest: do a per-row top-K.
            # We unmask roughly the same fraction across rows.
            # n_to_unmask is computed globally; per-row we unmask
            # ceil(n_to_unmask / B / 5) on average. Switch to per-row:
            n_per_row = max(1, int(round(n_to_unmask / B)))
            if step == n_steps - 1:
                n_per_row = total_masked_start // B + 1  # plenty

            # Per-row flat score and selection
            B_n = B
            # Stack flat scores per row: (B, n_R + n_F + n_L + n_Plen + n_Ppos)
            sR_flat = scores_R.reshape(B_n, -1)
            sF_flat = scores_F.reshape(B_n, -1)
            sL_flat = scores_L.reshape(B_n, -1)
            sPlen_flat = scores_Plen.reshape(B_n, -1)
            sPpos_flat = scores_Ppos.reshape(B_n, -1)
            sizes = [
                sR_flat.shape[1], sF_flat.shape[1], sL_flat.shape[1],
                sPlen_flat.shape[1], sPpos_flat.shape[1],
            ]
            cuts = [0]
            for s in sizes:
                cuts.append(cuts[-1] + s)
            big = torch.cat(
                [sR_flat, sF_flat, sL_flat, sPlen_flat, sPpos_flat], dim=1
            )  # (B, total)
            n_sel = min(n_per_row, big.shape[1])
            if n_sel <= 0:
                continue

            # Per-row top-K
            _, topk_idx = big.topk(n_sel, dim=1)  # (B, n_sel)

            # For each selected flat index, set the corresponding
            # element to its predicted class and clear maskedness.
            for r in range(B_n):
                for k in range(n_sel):
                    flat_idx = topk_idx[r, k].item()
                    # Scores set to -inf if masked=False, so any selected
                    # idx whose score is -inf shouldn't be processed.
                    score_val = big[r, flat_idx].item()
                    if not math.isfinite(score_val):
                        continue
                    if cuts[0] <= flat_idx < cuts[1]:
                        local = flat_idx - cuts[0]
                        R_t[r, local] = R_pred[r, local]
                        R_masked[r, local] = False
                    elif cuts[1] <= flat_idx < cuts[2]:
                        local = flat_idx - cuts[1]
                        a = local // self.r_max
                        b = local % self.r_max
                        F_t[r, a, b] = F_pred_full[r, a, b]
                        # mirror to lower
                        F_t[r, b, a] = F_pred_full[r, a, b]
                        F_masked[r, a, b] = False
                    elif cuts[2] <= flat_idx < cuts[3]:
                        local = flat_idx - cuts[2]
                        a = local // self.r_max
                        b = local % self.r_max
                        L_t[r, a, b] = L_pred_full[r, a, b]
                        L_t[r, b, a] = L_pred_full[r, a, b]
                        L_masked[r, a, b] = False
                    elif cuts[3] <= flat_idx < cuts[4]:
                        local = flat_idx - cuts[3]
                        a = local // self.p_max
                        b = local % self.p_max
                        Plen_t[r, a, b] = Plen_pred[r, a, b]
                        Plen_masked[r, a, b] = False
                    else:
                        local = flat_idx - cuts[4]
                        a = local // self.p_max
                        b = local % self.p_max
                        Ppos_t[r, a, b] = Ppos_pred[r, a, b]
                        Ppos_masked[r, a, b] = False

        # After last step, anything still masked gets argmax filled
        # (final-step n_to_unmask was set to clear all, but be defensive).
        any_remaining = (
            R_masked.any() or F_masked.any() or L_masked.any()
            or Plen_masked.any() or Ppos_masked.any()
        )
        if any_remaining:
            # Final greedy fill
            F_t_full = self._mirror_upper_to_full(F_t)
            L_t_full = self._mirror_upper_to_full(L_t)
            out = self.forward(
                R_t=R_t.where(~R_masked, torch.full_like(R_t, MASK_R)),
                F_full_t=F_t_full,
                L_full_t=L_t_full,
                Plen_t=Plen_t,
                Ppos_t=Ppos_t,
                alpha=torch.zeros(B, device=device),
                condition=condition,
            )
            R_pred = out["R_logits"].argmax(-1)
            F_pred_upper = out["F_logits"].argmax(-1)
            L_pred_upper = out["L_logits"].argmax(-1)
            Plen_pred = out["Plen_logits"].argmax(-1)
            Ppos_pred = out["Ppos_logits"].argmax(-1)
            R_t = torch.where(R_masked, R_pred, R_t)
            F_pred_full = torch.zeros_like(F_t)
            F_pred_full[:, self.pair_i, self.pair_j] = F_pred_upper
            F_pred_full[:, self.pair_j, self.pair_i] = F_pred_upper
            F_t = torch.where(F_masked, F_pred_full, F_t)
            L_pred_full = torch.zeros_like(L_t)
            L_pred_full[:, self.pair_i, self.pair_j] = L_pred_upper
            L_pred_full[:, self.pair_j, self.pair_i] = L_pred_upper
            L_t = torch.where(L_masked, L_pred_full, L_t)
            Plen_t = torch.where(Plen_masked, Plen_pred, Plen_t)
            Ppos_t = torch.where(Ppos_masked, Ppos_pred, Ppos_t)

        # Final F, L are already symmetric (mirroring during sampling)
        out_layout = {
            "R": R_t,
            "F": F_t,
            "L": L_t,
            "P_len": Plen_t,
            "P_pos": Ppos_t,
        }
        if post_process:
            out_layout = postprocess_layout(out_layout)
        return out_layout

    def _mirror_upper_to_full(self, mat: torch.Tensor) -> torch.Tensor:
        """Symmetrize: take upper triangle, copy to lower triangle.
        Used during sampling when L/F are partially built."""
        out = mat.clone()
        for ii in range(self.r_max):
            for jj in range(ii):
                # lower (ii, jj) ← upper (jj, ii)
                out[:, ii, jj] = out[:, jj, ii]
        return out


# ─── Post-processing ───────────────────────────────────────────────────

def postprocess_layout(
    layout: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Apply minimal cleanup so the layout is more likely to satisfy the
    decoder's structural constraints.

    Steps:
      (1) Mirror F upper → lower (symmetric); zero diagonal.
      (2) Zero L where F != 2 (linker only meaningful when linked).
          Zero L diagonal.
      (3) Zero P_pos where P_len == 0.
      (4) Left-pack R: re-order ring slots so non-PAD entries come first.
          When re-ordering R, the corresponding rows/cols of F, L, P_len,
          P_pos are also permuted to keep the layout consistent.

    Note: this does NOT enforce all decoder constraints (e.g., tree-shaped
    ring graph, anchor uniqueness, pendant-vs-fusion-edge collisions).
    Such violations cause decode_layout_to_scaffold to raise, which we
    treat as the rejection signal in evaluation.
    """
    R = layout["R"].clone()
    F_ = layout["F"].clone()
    L = layout["L"].clone()
    Plen = layout["P_len"].clone()
    Ppos = layout["P_pos"].clone()

    B = R.shape[0]
    r_max = R.shape[1]

    # (1) Symmetrize F via upper triangle (already symmetric in sample,
    #     but be defensive). Zero diagonal.
    diag_idx = torch.arange(r_max, device=R.device)
    F_[:, diag_idx, diag_idx] = 0
    upper = torch.triu(torch.ones(r_max, r_max, dtype=torch.bool,
                                   device=R.device), diagonal=1)
    F_upper_vals = F_ * upper.unsqueeze(0)
    F_ = F_upper_vals + F_upper_vals.transpose(-1, -2)

    # (2) Zero L where F != 2 (LINKED). Zero L diagonal.
    L[:, diag_idx, diag_idx] = 0
    L_upper_vals = L * upper.unsqueeze(0)
    L = L_upper_vals + L_upper_vals.transpose(-1, -2)
    L = torch.where(F_ == 2, L, torch.zeros_like(L))

    # (3) Zero P_pos where P_len == 0.
    Ppos = torch.where(Plen == 0, torch.zeros_like(Ppos), Ppos)

    # (4) Left-pack R per-row, re-permute F, L, Plen, Ppos.
    for b in range(B):
        order = []
        # First pass: indices of non-PAD rings in original order
        for k in range(r_max):
            if int(R[b, k]) != 0:
                order.append(k)
        # Then PAD slots
        for k in range(r_max):
            if int(R[b, k]) == 0:
                order.append(k)
        # Apply permutation
        perm = torch.tensor(order, dtype=torch.long, device=R.device)
        R[b] = R[b][perm]
        F_[b] = F_[b][perm][:, perm]
        L[b] = L[b][perm][:, perm]
        Plen[b] = Plen[b][perm]
        Ppos[b] = Ppos[b][perm]

    # After permutation, kill any P_len/P_pos in PAD ring slots
    pad_ring = (R == 0)  # (B, R_MAX)
    Plen = torch.where(
        pad_ring.unsqueeze(-1).expand_as(Plen),
        torch.zeros_like(Plen), Plen,
    )
    Ppos = torch.where(
        pad_ring.unsqueeze(-1).expand_as(Ppos),
        torch.zeros_like(Ppos), Ppos,
    )

    return {
        "R": R,
        "F": F_,
        "L": L,
        "P_len": Plen,
        "P_pos": Ppos,
    }


# ─── Capacity presets ──────────────────────────────────────────────────

CAPACITY_PRESETS: Dict[str, Dict] = {
    # (d_model, n_layers, n_heads, d_ff) — actual param counts on next line
    "600K": dict(d_model=96,  n_layers=4,  n_heads=4, d_ff=384),    # ~619K
    "1M":   dict(d_model=128, n_layers=4,  n_heads=4, d_ff=512),    # ~1.09M
    "3M":   dict(d_model=192, n_layers=6,  n_heads=4, d_ff=768),    # ~3.62M
    "10M":  dict(d_model=288, n_layers=8,  n_heads=8, d_ff=1152),   # ~10.6M
}


def build_ring_layout_diffusion(
    capacity: str = "1M",
    condition_dim: int = 2,
    cfg_drop_prob: float = 0.1,
    dropout: float = 0.1,
    schedule: str = "cosine",
    **overrides,
) -> RingLayoutDiffusion:
    """Build the A1 model at one of the named capacities.

    capacity ∈ {'600K', '1M', '3M', '10M'}.

    For larger datasets (ZINC250K, QM9), use '10M' or pass overrides
    like d_model=512, n_layers=12. The small layout vocab means A1
    rarely needs to be huge; A2 (atom assignment, Batch 4) is where
    capacity scales with molecular diversity.
    """
    if capacity not in CAPACITY_PRESETS:
        raise ValueError(
            f"Unknown capacity '{capacity}'. Options: "
            f"{list(CAPACITY_PRESETS.keys())}"
        )
    cfg = dict(CAPACITY_PRESETS[capacity])
    cfg.update(overrides)
    return RingLayoutDiffusion(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        condition_dim=condition_dim,
        cfg_drop_prob=cfg_drop_prob,
        dropout=dropout,
        schedule=schedule,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
