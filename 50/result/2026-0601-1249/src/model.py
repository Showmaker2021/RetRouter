"""
RetNet-NCO: Retentive Network for Neural Combinatorial Optimization
Replaces CADA's Transformer encoder/decoder with:
  - Encoder: RetNet (parallel mode training) with spatial decay kernel
  - Decoder: Hierarchical Route Memory (S_local + S_global) with boundary-aware gating

Key design decisions:
  - Encoder uses chunkwise retention in parallel mode (training efficiency)
  - Spatial decay: gamma learned per head, initialized from Euclidean distance prior
  - S_local resets at every depot visit; S_global accumulates across sub-tours
  - Boundary injection: when S_local resets, depot embedding is added to S_global
  - Cross-attention (decoder → encoder outputs) kept as standard attention (fixed size)
  - All other interfaces (PrecomputedCache, VRPModel.forward) unchanged for trainer compatibility
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Union, Optional
from dataclasses import dataclass, fields
from tensordict import TensorDict
from torch import Tensor

from utils.functions import batchify, gather_by_index, unbatchify, unbatchify_and_gather


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def linear_layer(input_dim, output_dim, std=1e-2, bias=True):
    linear = nn.Linear(input_dim, output_dim, bias=bias)
    nn.init.normal_(linear.weight, std=std)
    nn.init.zeros_(linear.bias)
    return linear


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class AddAndNorm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_type = model_params['norm_type']
        if self.norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif self.norm_type == 'layer':
            self.norm = nn.LayerNorm(embedding_dim)
        elif self.norm_type == 'rms':
            self.norm = RMSNorm(embedding_dim)
        else:
            raise NotImplementedError

    def forward(self, input1, input2):
        added = input1 + input2
        if self.norm_type == 'instance':
            return self.norm(added.transpose(1, 2)).transpose(1, 2)
        return self.norm(added)


# ---------------------------------------------------------------------------
# FFN variants (kept from original)
# ---------------------------------------------------------------------------

class ParallelGatedMLP(nn.Module):
    def __init__(self, hidden_size=128, inner_size_multiple_of=256,
                 mlp_activation="silu", model_parallel_size=1):
        super().__init__()
        self.act = F.silu if mlp_activation == "silu" else F.gelu
        multiple_of = inner_size_multiple_of * model_parallel_size
        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = multiple_of * ((inner_size + multiple_of - 1) // multiple_of)
        self.l1 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, z):
        return self.l3(self.act(self.l1(z)) * self.l2(z))


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))


# ---------------------------------------------------------------------------
# PromptNet (unchanged)
# ---------------------------------------------------------------------------

class PromptNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = 5
        output_dim = args.model_params['embedding_dim']
        self.logit_clipping = args.model_params['logit_clipping']
        layer1 = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.uniform_(layer1.weight)
        self.model = nn.Sequential(
            layer1,
            nn.LayerNorm(output_dim),
            linear_layer(output_dim, output_dim),
            nn.ReLU(),
            linear_layer(output_dim, output_dim // 8),
            nn.LayerNorm(output_dim // 8),
            linear_layer(output_dim // 8, 5 * output_dim),
        )

    def forward(self, td):
        return {"prompt": self.model(td['p_s_tag'][:, :5]).view(td.batch_size[0], 5, -1)}


# ---------------------------------------------------------------------------
# PrecomputedCache (unchanged interface)
# ---------------------------------------------------------------------------

@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)


# ===========================================================================
# RETNET ENCODER
# ===========================================================================

class RetNetEncoderLayer(nn.Module):
    """
    One RetNet encoder layer (parallel mode).

    Retention in parallel mode:
        Retention(X) = (Q K^T ⊙ D) V
    where D_ij = gamma^(i-j) if i >= j else 0  (causal masking)

    For NCO encoder, we use BIDIRECTIONAL retention:
        D_ij = gamma^|i-j|  (symmetric decay by sequence distance)

    With spatial coords, gamma is replaced by a spatial decay:
        D_ij = exp(-lambda * ||coord_i - coord_j||)
    where lambda is a learned per-head parameter.

    Training: parallel mode — O(N²) but fully parallelizable (like Transformer)
    The recurrent form is used only in the decoder (not encoder).
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        self.head_num = head_num
        self.qkv_dim = qkv_dim

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.group_norm = nn.GroupNorm(head_num, head_num * qkv_dim)
        self.W_out = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Spatial decay: lambda per head, init from config
        # gamma(i,j) = exp(-lambda_h * dist(i,j)), lambda = exp(log_lambda) > 0
        log_lambda_init = model_params.get('retnet_log_lambda_init', 0.0)
        self.log_lambda = nn.Parameter(
            torch.full((head_num,), log_lambda_init)
        )

        # Norm + FFN
        self.norm1 = RMSNorm(embedding_dim)
        self.norm2 = RMSNorm(embedding_dim)
        if model_params.get('ffd', 'siglu') == 'siglu':
            self.ffn = ParallelGatedMLP(hidden_size=embedding_dim)
        else:
            self.ffn = FeedForward(**model_params)

    def _spatial_decay_matrix(self, coords: Tensor) -> Tensor:
        """
        Compute spatial decay matrix D of shape (batch, head_num, N, N).

        D[b, h, i, j] = exp(-lambda_h * ||coord_i - coord_j||_2)

        coords: (batch, N, 2)
        """
        # Pairwise Euclidean distance: (batch, N, N)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)          # (B, N, N, 2)
        dist = torch.norm(diff, dim=-1)                            # (B, N, N)

        # lambda per head: (head_num,) → (1, H, 1, 1)
        lam = self.log_lambda.exp().view(1, self.head_num, 1, 1)  # always positive
        dist_expanded = dist.unsqueeze(1)                          # (B, 1, N, N)
        D = torch.exp(-lam * dist_expanded)                        # (B, H, N, N)
        return D

    def _sequence_decay_matrix(self, N: int, device) -> Tensor:
        """
        Fallback: sequence-distance decay when coords not available.
        D[i, j] = gamma^|i-j| where gamma = sigmoid(log_lambda) ∈ (0,1)
        Shape: (1, head_num, N, N)
        """
        idx = torch.arange(N, device=device, dtype=torch.float)
        dist_seq = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()    # (N, N)
        gamma = torch.sigmoid(self.log_lambda).view(1, self.head_num, 1, 1)
        D = gamma ** dist_seq.unsqueeze(0).unsqueeze(0)            # (1, H, N, N)
        return D

    def forward(self, x: Tensor, coords: Optional[Tensor] = None) -> Tensor:
        """
        x: (batch, N, embedding_dim)
        coords: (batch, N, 2) — node coordinates for spatial decay
        """
        B, N, _ = x.shape
        residual = x
        x = self.norm1(x)

        # QKV
        q = reshape_by_heads(self.Wq(x), self.head_num)  # (B, H, N, qkv_dim)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)

        # Decay matrix
        if coords is not None:
            D = self._spatial_decay_matrix(coords)  # (B, H, N, N)
        else:
            D = self._sequence_decay_matrix(N, x.device)  # (1, H, N, N)

        # Parallel retention: (Q K^T ⊙ D) V
        scale = self.qkv_dim ** -0.5
        qk = torch.matmul(q, k.transpose(-2, -1)) * scale        # (B, H, N, N)
        retention = (qk * D)                                       # (B, H, N, N)

        # Normalize retention (GroupNorm variant from RetNet paper)
        out = torch.matmul(retention, v)                           # (B, H, N, qkv_dim)
        out = out.transpose(1, 2).reshape(B, N, self.head_num * self.qkv_dim)
        # GroupNorm along head*qkv_dim dim
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.W_out(out)                                      # (B, N, embed)

        # Residual + FFN
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


# ===========================================================================
# RETNET ENCODER (full model)
# ===========================================================================

class VRP_Encoder(nn.Module):
    """
    CADA encoder replaced with RetNet encoder.
    Keeps dual-branch (sparse + global) structure from original CADA
    but both branches use RetNetEncoderLayer instead of Transformer layers.

    Branch 1 (out):  spatial decay retention — captures local structure
    Branch 2 (out2): sequence-distance retention + prompt — captures global
    Combined via learned linear projections (same as CADA).
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']
        self.p_num = model_params['p_num']

        # Input projections (same as original)
        self.embedding_depot = nn.Linear(3, embedding_dim)
        self.embedding_node = nn.Linear(7, embedding_dim)

        # Branch 1: spatial RetNet layers
        self.layers = nn.ModuleList([
            RetNetEncoderLayer(**model_params) for _ in range(encoder_layer_num)
        ])

        # Branch 2: global RetNet layers (no sparse — global attention equivalent)
        model_params_global = model_params.copy()
        model_params_global['use_sparse'] = False
        self.layers2 = nn.ModuleList([
            RetNetEncoderLayer(**model_params_global) for _ in range(encoder_layer_num)
        ])

        # Cross-branch combination (same as CADA)
        self.layers1combine = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(encoder_layer_num)
        ])
        self.layers2combine = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(encoder_layer_num - 1)
        ])

    def forward(self, td, prompt):
        """
        Returns: (batch, N+1, embedding_dim)
        """
        # --- Build features (identical to original CADA) ---
        depot_feats = torch.cat(
            [td["locs"][:, :1, :], td["distance_limit"][..., None]], -1
        )  # (B, 1, 3)
        node_feats = torch.cat(
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["locs"][:, 1:, :],
            ), -1,
        )  # (B, N, 7)
        depot_feats = torch.nan_to_num(depot_feats, nan=0.0, posinf=0.0, neginf=0.0)
        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)

        bs, n, _ = node_feats.shape
        global_embeddings = self.embedding_depot(depot_feats)  # (B, 1, D)
        cust_embeddings = self.embedding_node(node_feats)       # (B, N, D)
        out = torch.cat((global_embeddings, cust_embeddings), dim=1)  # (B, N+1, D)
        out2 = out

        # Spatial coords for decay: (B, N+1, 2)
        coords = td["locs"]  # depot at index 0, customers at 1..N

        for i, layer in enumerate(self.layers):
            if i == 0:
                out2 = torch.cat((out2, prompt), dim=1)  # (B, N+1+p_num, D)

            # Branch 1: spatial retention
            out = layer(out, coords=coords)

            # Branch 2: global retention (no spatial coords → sequence decay)
            # coords2 includes prompt tokens — use None for sequence decay
            out2 = self.layers2[i](out2, coords=None)

            # Combine branch2 → branch1
            out = out + self.layers1combine[i](out2[:, :n + 1])

            if i != len(self.layers) - 1:
                out2_ = out2[:, :n + 1] + self.layers2combine[i](out)
                out2_ = torch.cat((out2_, out2[:, -self.p_num:]), dim=1)
                out2 = out2_

        return out[:, :n + 1]  # (B, N+1, D)


# ===========================================================================
# HIERARCHICAL ROUTE MEMORY DECODER
# ===========================================================================

class HierarchicalRouteMemory(nn.Module):
    """
    Two-level recurrent retention state for the NCO decoder.

    S_local[h]:  (B, num_starts, qkv_dim, qkv_dim) per head
        - Reset to 0 every time the vehicle returns to the depot
        - Captures the pattern within the current sub-tour

    S_global[h]: (B, num_starts, qkv_dim, qkv_dim) per head
        - Never reset; slow exponential decay (γ_global < γ_local)
        - Captures patterns across all sub-tours
        - Receives boundary injection when a sub-tour ends

    Update rule (recurrent RetNet):
        S_local  ← γ_local  · S_local  + k ⊗ v   (zero if at depot)
        S_global ← γ_global · S_global + k ⊗ v
                   + W_boundary · depot_embed     (if at depot)

    Retrieval:
        y_local  = q · S_local      (B, num_starts, H, qkv_dim)
        y_global = q · S_global
        gate = sigmoid(W_gate · cat(y_local_flat, y_global_flat, q_raw))
        y = gate * y_local + (1-gate) * y_global

    Cross-attention into encoder outputs is kept as standard attention
    (encoder output is fixed-size N+1 — no complexity advantage of recurrence there).
    """

    def __init__(self, model_params: dict):
        super().__init__()
        head_num     = model_params['head_num']
        qkv_dim      = model_params['qkv_dim']
        embedding_dim = model_params['embedding_dim']
        self.head_num     = head_num
        self.qkv_dim      = qkv_dim
        self.embedding_dim = embedding_dim

        # --- Decay parameters (learned, init from config) ---
        # gamma = sigmoid(logit), so logit = log(p/(1-p))
        gamma_local_init  = model_params.get('hrm_gamma_local_logit_init',  2.197)  # → 0.90
        gamma_global_init = model_params.get('hrm_gamma_global_logit_init', 4.595)  # → 0.99
        self.gamma_local  = nn.Parameter(torch.full((head_num,), gamma_local_init))
        self.gamma_global = nn.Parameter(torch.full((head_num,), gamma_global_init))

        # --- Boundary injection: depot_embed → S_global ---
        boundary_std = model_params.get('hrm_boundary_std', 0.01)
        self.W_boundary = nn.Linear(embedding_dim, head_num * qkv_dim * qkv_dim, bias=False)
        nn.init.normal_(self.W_boundary.weight, std=boundary_std)

        # --- Gating: local vs global ---
        # input: [y_local | y_global | q_raw]  all of size H*D
        gate_input_dim = head_num * qkv_dim * 3
        self.W_gate = nn.Linear(gate_input_dim, head_num * qkv_dim, bias=True)
        nn.init.zeros_(self.W_gate.weight)
        gate_bias_init = model_params.get('hrm_gate_bias_init', 0.0)
        nn.init.constant_(self.W_gate.bias, gate_bias_init)

        # Output projection after memory retrieval
        self.W_out = nn.Linear(head_num * qkv_dim, embedding_dim)

        # GroupNorm for retention output (from RetNet paper)
        self.group_norm = nn.GroupNorm(head_num, head_num * qkv_dim)

    def init_states(self, batch_size: int, num_starts: int, device) -> Tuple[Tensor, Tensor]:
        """Initialize S_local and S_global to zero."""
        H, D = self.head_num, self.qkv_dim
        shape = (batch_size, num_starts, H, D, D)
        S_local = torch.zeros(shape, device=device)
        S_global = torch.zeros(shape, device=device)
        return S_local, S_global

    def forward(
        self,
        q: Tensor,           # (B, num_starts, H, qkv_dim) — decoder query
        k: Tensor,           # (B, num_starts, H, qkv_dim) — key from current node
        v: Tensor,           # (B, num_starts, H, qkv_dim) — value from current node
        S_local: Tensor,     # (B, num_starts, H, qkv_dim, qkv_dim)
        S_global: Tensor,    # (B, num_starts, H, qkv_dim, qkv_dim)
        at_depot: Tensor,    # (B, num_starts) bool — True if current action = depot
        depot_embed: Tensor, # (B, 1, embedding_dim) — depot node embedding from encoder
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            out:      (B, num_starts, embedding_dim) — memory retrieval output
            S_local:  updated local state
            S_global: updated global state
        """
        B, S, H, D = q.shape
        gl = torch.sigmoid(self.gamma_local).view(1, 1, H, 1, 1)   # (1,1,H,1,1)
        gg = torch.sigmoid(self.gamma_global).view(1, 1, H, 1, 1)

        # outer product k ⊗ v: (B, S, H, D, D)
        kv = torch.einsum('bshd,bshe->bshde', k, v)

        # ---- Update S_local ----
        # reset to 0 at depot; then accumulate
        at_depot_expanded = at_depot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,S,1,1,1)
        S_local_decay = gl * S_local + kv
        S_local_new = torch.where(at_depot_expanded, torch.zeros_like(S_local), S_local_decay)

        # ---- Update S_global ----
        # Boundary injection: when returning to depot, add depot signal
        depot_embed_flat = depot_embed.squeeze(1)                             # (B, embedding_dim)
        boundary_signal = self.W_boundary(depot_embed_flat)                   # (B, H*D*D)
        boundary_signal = boundary_signal.view(B, 1, H, D, D)               # (B,1,H,D,D)
        boundary_signal = boundary_signal.expand(B, S, H, D, D)

        S_global_decay = gg * S_global + kv
        S_global_new = torch.where(
            at_depot_expanded,
            S_global_decay + boundary_signal,  # inject boundary event
            S_global_decay,
        )

        # ---- Retrieve from memory ----
        # y = q · S: (B, S, H, 1, D) @ (B, S, H, D, D)^-like
        # q shape: (B, S, H, D) → unsqueeze → (B, S, H, 1, D)
        q_unsq = q.unsqueeze(-2)
        y_local = torch.matmul(q_unsq, S_local_new).squeeze(-2)    # (B, S, H, D)
        y_global = torch.matmul(q_unsq, S_global_new).squeeze(-2)  # (B, S, H, D)

        # Flatten for gating
        y_local_flat = y_local.reshape(B, S, H * D)
        y_global_flat = y_global.reshape(B, S, H * D)
        q_flat = q.reshape(B, S, H * D)

        gate_input = torch.cat([y_local_flat, y_global_flat, q_flat], dim=-1)
        gate = torch.sigmoid(self.W_gate(gate_input))              # (B, S, H*D)

        y_merged = gate * y_local_flat + (1 - gate) * y_global_flat  # (B, S, H*D)

        # GroupNorm
        y_norm = self.group_norm(
            y_merged.reshape(B * S, H * D, 1)
        ).reshape(B, S, H * D)

        out = self.W_out(y_norm)  # (B, S, embedding_dim)

        return out, S_local_new, S_global_new


class VRP_Decoder(nn.Module):
    """
    Decoder with Hierarchical Route Memory.

    Per decoding step:
    1. Build context query from [current_node_embed | state_features]
    2. Project to Q, K, V
    3. Update S_local / S_global via HierarchicalRouteMemory
    4. Retrieve y_memory from hierarchical memory
    5. Cross-attention: y_memory queries into full encoder output (standard MHA)
       → this is O(N) per step but N is fixed (encoder output doesn't grow)
    6. Pointer: single-head dot-product into encoder output for logits
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        # Context projection: [node_embed (D) | state (5)] → query
        self.Wq_context = nn.Linear(embedding_dim + 5, head_num * qkv_dim, bias=False)

        # Key / Value projections for retention update (from current node embed)
        self.Wk_ret = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_ret = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # Hierarchical route memory
        self.hrm = HierarchicalRouteMemory(model_params)

        # Cross-attention: memory output → encoder outputs (standard MHA)
        # (encoder output is fixed, so we keep standard attention here)
        self.Wq_cross = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_cross = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_cross = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.cross_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Pointer (single-head dot-product into encoder output)
        # logit_key = encoder_out.T — same as original CADA
        self.norm = RMSNorm(embedding_dim)

        # Precompute cross-attention K/V from encoder (cached)
        self.Wk_enc = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_enc = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

    def init_memory(self, batch_size: int, num_starts: int, device) -> Tuple[Tensor, Tensor]:
        return self.hrm.init_states(batch_size, num_starts, device)

    def forward(
        self,
        td,
        cache: PrecomputedCache,
        num_starts: int,
        S_local: Tensor,
        S_global: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        td is already flat: (B*num_starts, ...)
        cache tensors are already flat: (B*num_starts, ...)
        S_local / S_global shape: (B, num_starts, H, D, D)

        Returns:
            log_probs:  (B*num_starts, N+1)
            mask:       (B*num_starts, N+1)
            S_local:    updated (B, num_starts, H, D, D)
            S_global:   updated
        """
        H  = self.model_params['head_num']
        D  = self.model_params['qkv_dim']
        BS = td.batch_size[0]          # B * num_starts (flat)
        B  = BS // num_starts

        # --- All tensors in flat (BS, ...) space ---
        # Current node embedding: (BS, embed_dim)
        cur_node_embed = gather_by_index(cache.node_embeddings, td["current_node"])

        # State features: (BS, 5)
        rl    = td["vehicle_capacity"] - td["used_capacity_linehaul"]
        rb    = td["vehicle_capacity"] - td["used_capacity_backhaul"]
        state = torch.cat([rl, rb, td["current_time"],
                           td["current_route_length"], td["open_route"]], -1)

        context = torch.cat([cur_node_embed, state], -1)  # (BS, D+5)

        # --- Retention Q/K/V  (BS, H, D) → reshape to (B, S, H, D) for HRM ---
        q_flat = self.Wq_context(context).reshape(BS, H, D)   # (BS, H, D)
        k_flat = self.Wk_ret(cur_node_embed).reshape(BS, H, D)
        v_flat = self.Wv_ret(cur_node_embed).reshape(BS, H, D)

        # Reshape to (B, num_starts, H, D) for hierarchical memory
        q_ret = q_flat.view(B, num_starts, H, D)
        k_ret = k_flat.view(B, num_starts, H, D)
        v_ret = v_flat.view(B, num_starts, H, D)

        # Detect depot visits: (BS,) → (B, num_starts)
        at_depot = (td["current_node"].squeeze(-1) == 0).view(B, num_starts)

        # Depot embedding from encoder output node 0: (BS, 1, embed_dim)
        # Reduce to (B, 1, embed_dim) — depot is shared across starts
        depot_embed = cache.node_embeddings[:, :1, :].view(B, num_starts, 1, -1)[:, 0, :, :]
        # shape: (B, 1, embed_dim)

        # --- Hierarchical Route Memory ---
        memory_out, S_local, S_global = self.hrm(
            q_ret, k_ret, v_ret, S_local, S_global, at_depot, depot_embed
        )
        # memory_out: (B, num_starts, embed_dim) → flatten back
        memory_out_flat = memory_out.view(BS, -1)  # (BS, embed_dim)

        # --- Cross-attention: memory → encoder nodes ---
        # cache.glimpse_key: (BS, H, N+1, D)
        mask = td["action_mask"]  # (BS, N+1)

        # q_cross: (BS, 1, H, D) to match k/v shape for batched cross-attention
        q_cross = self.Wq_cross(memory_out_flat).reshape(BS, H, 1, D)  # (BS, H, 1, D)
        # glimpse_key already (BS, H, N+1, D)
        cross_out = _multi_head_cross_attention_flat(
            q_cross, cache.glimpse_key, cache.glimpse_val, mask
        )  # (BS, embed_dim)

        mh_out = self.cross_combine(cross_out)           # (BS, embed_dim)
        mh_out = self.norm(mh_out + memory_out_flat)     # (BS, embed_dim)

        # --- Pointer ---
        # cache.logit_key: (BS, embed_dim, N+1)
        score = torch.matmul(mh_out.unsqueeze(1), cache.logit_key)  # (BS, 1, N+1)
        score = score.squeeze(1) / self.model_params['sqrt_embedding_dim']  # (BS, N+1)

        logits = torch.tanh(score) * self.model_params['logit_clipping']
        logits[~mask] = float("-inf")

        return F.log_softmax(logits, dim=-1), mask, S_local, S_global


def _multi_head_cross_attention_flat(q, k, v, mask=None):
    """
    Cross-attention, all tensors in flat batch dimension.
    q: (BS, H, 1, D)      — one query per instance
    k: (BS, H, N+1, D)
    v: (BS, H, N+1, D)
    mask: (BS, N+1)  bool — True = valid action
    Returns: (BS, H*D)
    """
    BS, H, _, D = q.shape
    N1 = k.size(2)
    scale = D ** -0.5
    score = torch.matmul(q, k.transpose(-2, -1)) * scale  # (BS, H, 1, N+1)
    if mask is not None:
        # mask: (BS, N+1) → (BS, 1, 1, N+1)
        score = score.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    weights = torch.softmax(score, dim=-1)                # (BS, H, 1, N+1)
    out = torch.matmul(weights, v)                        # (BS, H, 1, D)
    out = out.squeeze(2).reshape(BS, H * D)               # (BS, H*D)
    return out


# ===========================================================================
# MAIN MODEL
# ===========================================================================

class VRPModel(nn.Module):
    """
    RetNet-NCO model.
    Encoder: RetNet (parallel mode, spatial decay)
    Decoder: Hierarchical Route Memory (recurrent retention)

    Interface unchanged from original CADA VRPModel.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = VRP_Encoder(**args.model_params)
        self.decoder = VRP_Decoder(**args.model_params)
        self.prompt_net = PromptNet(args)

    @staticmethod
    def greedy(logprobs, mask=None):
        selected = logprobs.argmax(dim=-1)
        if mask is not None:
            assert not (~mask).gather(1, selected.unsqueeze(-1)).data.any(), \
                "infeasible action selected"
        return selected

    @staticmethod
    def sampling(logprobs, log, mask=None):
        probs = logprobs.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)
        if mask is not None:
            while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
                log("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
        return selected

    def forward(self, td, env):
        args = self.args

        # --- Encode ---
        p_out = self.prompt_net(td)
        prompt = p_out['prompt']
        node_embed = self.encoder(td, prompt)  # (B, N+1, D)

        # --- Multi-start setup ---
        num_starts, action = env.select_start_nodes(td)
        td = batchify(td, num_starts)
        logprobs_list = [torch.zeros_like(action, device=td.device)]
        actions_list = [action]

        td.set("action", action)
        td = env.step(td)["next"]

        # Precompute cross-attention K/V from encoder, then batchify for multi-start
        head_num = args.model_params['head_num']
        decoder_k         = reshape_by_heads(self.decoder.Wk_enc(node_embed), head_num=head_num)
        decoder_v         = reshape_by_heads(self.decoder.Wv_enc(node_embed), head_num=head_num)
        decoder_logit_key = node_embed.transpose(1, 2)  # (B, D, N+1)

        # Batchify once — decoder always receives flat (B*num_starts, ...) tensors
        cache = PrecomputedCache(
            batchify(node_embed,         num_starts),
            batchify(decoder_k,          num_starts),
            batchify(decoder_v,          num_starts),
            batchify(decoder_logit_key,  num_starts),
        )

        # Initialize hierarchical memory: (B_orig, num_starts, H, qkv_dim, qkv_dim)
        B_orig = node_embed.size(0)
        S_local, S_global = self.decoder.init_memory(
            B_orig, num_starts, node_embed.device
        )

        # --- Decoding loop ---
        while not td["done"].all():
            logprobs, mask, S_local, S_global = self.decoder(
                td, cache, num_starts, S_local, S_global
            )
            if self.training:
                select = VRPModel.sampling(logprobs, args.log, mask)
            else:
                select = VRPModel.greedy(logprobs, mask)

            logprobs = gather_by_index(logprobs, select, dim=1)
            td.set("action", select)
            actions_list.append(select)
            logprobs_list.append(logprobs)
            td = env.step(td)["next"]

        # --- Post-process ---
        logprobs = torch.stack(logprobs_list, 1)
        actions = torch.stack(actions_list, 1)
        td.set("reward", env.get_reward(td, actions))
        assert (logprobs > -1000).data.all(), "Logprobs should not be -inf!"
        log_likelihood_sum = logprobs.sum(1)

        return {
            "reward": td["reward"],
            "log_likelihood": log_likelihood_sum,
        }