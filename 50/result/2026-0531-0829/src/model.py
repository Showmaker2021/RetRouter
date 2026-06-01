"""
HRM-NCO: Hierarchical Route Memory for Neural Combinatorial Optimization
========================================================================

Three original contributions (no dependency on CADA architecture):

1. RETENTION ENCODER with Spatial-Semantic Decay Kernel
   - Bidirectional retention: Ret(X) = (Q K^T ⊙ D) V
   - D[b,h,i,j] = exp(-λ_h · f_θ(coord_i, coord_j, feat_i, feat_j))
   - f_θ: small MLP learning asymmetric spatial-semantic affinity
   - λ_h: learned per-head decay rate
   - Standard pre-norm Transformer structure (pure RetNet, no CADA dual-branch)

2. HIERARCHICAL ROUTE MEMORY (HRM)
   - S_local:  recurrent state, resets at every depot visit
               captures "what am I doing in this sub-tour"
   - S_global: slow-decay recurrent state, never resets
               captures "what patterns have I seen across all sub-tours"
   - Boundary injection: depot embedding added to S_global at sub-tour end
   - Gating: learned combination of local vs global memory per step

3. DYNAMIC TASK MEMORY (DTM) — replaces static prompt conditioning
   - Task embedding E_task from constraint flags (C/O/B/L/TW)
   - At each step: task_ctx = CrossAttn(S_global, E_task)
     → "given my tour history, which constraints are currently binding?"
   - task_ctx modulates decoder query: q = q_base + W_task · task_ctx
   - This is DYNAMIC: task conditioning changes per step based on memory state
   - Contrast with CADA/MTPOMO: static prompt injected once into encoder

Architecture:
    Input → [Retention Encoder] → Node Embeddings
          → [HRM Decoder with DTM] → Actions
             ↑ S_local, S_global maintained across decoding steps
             ↑ E_task computed once, modulates query dynamically each step
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass, fields
from tensordict import TensorDict
from torch import Tensor

from utils.functions import batchify, gather_by_index, unbatchify


# ============================================================================
# UTILITY
# ============================================================================

def linear_layer(in_dim, out_dim, std=1e-2, bias=True):
    l = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.normal_(l.weight, std=std)
    if bias:
        nn.init.zeros_(l.bias)
    return l


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


class ParallelGatedMLP(nn.Module):
    """SiLU-gated MLP (SwiGLU variant). embedding_dim=128 → inner=512."""
    def __init__(self, dim: int = 128, multiple_of: int = 256):
        super().__init__()
        inner = int(2 * dim * 4 / 3)
        inner = multiple_of * ((inner + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ============================================================================
# PRECOMPUTED CACHE
# ============================================================================

@dataclass
class PrecomputedCache:
    node_embeddings: Tensor   # (BS, N+1, D)
    glimpse_key:     Tensor   # (BS, H, N+1, qkv_dim)
    glimpse_val:     Tensor   # (BS, H, N+1, qkv_dim)
    logit_key:       Tensor   # (BS, D, N+1)

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts: int) -> "PrecomputedCache":
        return PrecomputedCache(*[batchify(f, num_starts) for f in self.fields])


# ============================================================================
# 1. SPATIAL-SEMANTIC DECAY KERNEL
# ============================================================================

class SpatialSemanticDecay(nn.Module):
    """
    Asymmetric decay kernel combining spatial distance and node features.

    D[b, h, i, j] = exp(-λ_h · f_θ(coord_i, coord_j, feat_i, feat_j))

    f_θ is a 2-layer MLP mapping pairwise features to a scalar affinity.
    Input to f_θ: [coord_i | coord_j | feat_i | feat_j | coord_i - coord_j]
    where feat = [demand_linehaul, demand_backhaul, tw_start, tw_end, service_time]

    Asymmetric: D[i→j] ≠ D[j→i] because time windows create directional constraints.
    """

    def __init__(self, coord_dim: int = 2, feat_dim: int = 5, head_num: int = 8,
                 hidden_dim: int = 32, log_lambda_init: float = 0.0):
        super().__init__()
        self.head_num = head_num
        # Input: coord_i(2) + coord_j(2) + feat_i(5) + feat_j(5) + delta_coord(2) = 16
        in_dim = coord_dim * 3 + feat_dim * 2  # 2+2+5+5+2 = 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Softplus(),   # output > 0, so decay is always < 1
        )
        nn.init.normal_(self.mlp[0].weight, std=0.01)
        nn.init.normal_(self.mlp[2].weight, std=0.01)
        # Per-head λ: scalar scale of the affinity output
        self.log_lambda = nn.Parameter(torch.full((head_num,), log_lambda_init))

    def forward(self, coords: Tensor, node_feats: Tensor) -> Tensor:
        """
        coords:     (B, N, 2)
        node_feats: (B, N, 5)  — [dl, db, tw_s, tw_e, st]
        Returns D:  (B, H, N, N)
        """
        B, N, _ = coords.shape
        # Pairwise feature construction: (B, N, N, in_dim)
        ci = coords.unsqueeze(2).expand(B, N, N, 2)        # coord_i repeated over j
        cj = coords.unsqueeze(1).expand(B, N, N, 2)        # coord_j repeated over i
        fi = node_feats.unsqueeze(2).expand(B, N, N, 5)
        fj = node_feats.unsqueeze(1).expand(B, N, N, 5)
        delta = ci - cj                                      # directional offset i→j

        pair = torch.cat([ci, cj, fi, fj, delta], dim=-1)  # (B, N, N, 16)
        affinity = self.mlp(pair).squeeze(-1)                # (B, N, N)

        # Per-head λ scaling
        lam = self.log_lambda.exp().view(1, self.head_num, 1, 1)   # (1,H,1,1)
        D = torch.exp(-lam * affinity.unsqueeze(1))                 # (B,H,N,N)
        return D


# ============================================================================
# 2. RETENTION ENCODER LAYER
# ============================================================================

class RetNetEncoderLayer(nn.Module):
    """
    Single bidirectional retention layer.

    Parallel mode (training):
        Ret(X) = (Q K^T ⊙ D) V
    GroupNorm normalization as in original RetNet paper.
    Pre-norm with RMSNorm.
    """

    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: int,
                 decay_kernel: Optional[SpatialSemanticDecay] = None, **kwargs):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim  = qkv_dim
        self.scale    = qkv_dim ** -0.5

        self.norm1 = RMSNorm(embedding_dim)
        self.norm2 = RMSNorm(embedding_dim)

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.group_norm = nn.GroupNorm(head_num, head_num * qkv_dim)
        self.W_out      = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.ffn = ParallelGatedMLP(dim=embedding_dim)

        # Optional: shared decay kernel (None → use sequence-distance fallback)
        self.decay_kernel = decay_kernel

        # Fallback decay when kernel not provided: γ^|i-j|, γ per head
        self.log_gamma = nn.Parameter(torch.zeros(head_num))  # sigmoid → ~0.5 init

    def _fallback_decay(self, N: int, device) -> Tensor:
        """Sequence-distance decay: (1, H, N, N)"""
        idx = torch.arange(N, device=device, dtype=torch.float)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        gamma = torch.sigmoid(self.log_gamma).view(1, self.head_num, 1, 1)
        return gamma ** dist.unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor, D: Optional[Tensor] = None) -> Tensor:
        """
        x: (B, N, D_emb)
        D: (B, H, N, N) decay matrix — precomputed outside to share across layers
        """
        B, N, _ = x.shape
        residual = x
        x = self.norm1(x)

        # QKV: reshape to (B, H, N, qkv_dim)
        q = self.Wq(x).reshape(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        k = self.Wk(x).reshape(B, N, self.head_num, self.qkv_dim).transpose(1, 2)
        v = self.Wv(x).reshape(B, N, self.head_num, self.qkv_dim).transpose(1, 2)

        if D is None:
            D = self._fallback_decay(N, x.device)

        # Parallel retention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        ret  = torch.matmul(attn * D, v)                           # (B,H,N,qkv)

        # GroupNorm + project
        out = ret.transpose(1, 2).reshape(B, N, self.head_num * self.qkv_dim)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.W_out(out)

        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# 3. VRP ENCODER (pure RetNet, no CADA dual-branch)
# ============================================================================

class VRP_Encoder(nn.Module):
    """
    Pure RetNet encoder for multi-task VRP.
    - Standard stack of L RetNetEncoderLayers (pre-norm)
    - Shared SpatialSemanticDecay kernel computed once per forward pass,
      reused across all layers (saves compute, encourages consistent decay)
    - No dual-branch, no prompt injection — task info goes into DTM
    """

    def __init__(self, **mp):
        super().__init__()
        D   = mp['embedding_dim']
        H   = mp['head_num']
        qkv = mp['qkv_dim']
        L   = mp['encoder_layer_num']
        log_lam_init = mp.get('retnet_log_lambda_init', 0.0)

        # Input projections
        # depot: (locs(2), distance_limit(1)) → 3 features
        # node:  (dl(1), db(1), tw(2), st(1), locs(2)) → 7 features
        self.embed_depot = nn.Linear(3, D)
        self.embed_node  = nn.Linear(7, D)

        # Shared spatial-semantic decay kernel
        self.decay_kernel = SpatialSemanticDecay(
            coord_dim=2, feat_dim=5, head_num=H,
            hidden_dim=mp.get('decay_hidden_dim', 32),
            log_lambda_init=log_lam_init,
        )

        # Encoder stack — all layers share the same decay_kernel object
        self.layers = nn.ModuleList([
            RetNetEncoderLayer(D, H, qkv, decay_kernel=self.decay_kernel)
            for _ in range(L)
        ])

        self.final_norm = RMSNorm(D)

    def forward(self, td: TensorDict) -> Tensor:
        """Returns: (B, N+1, D)"""
        # Build node features
        depot_feat = torch.cat(
            [td["locs"][:, :1, :], td["distance_limit"][..., None]], dim=-1
        )  # (B, 1, 3)

        node_feat = torch.cat([
            td["demand_linehaul"][..., 1:, None],   # (B, N, 1)
            td["demand_backhaul"][..., 1:, None],
            td["time_windows"][..., 1:, :],          # (B, N, 2)
            td["service_time"][..., 1:, None],
            td["locs"][:, 1:, :],                    # (B, N, 2)
        ], dim=-1)  # (B, N, 7)

        depot_feat = torch.nan_to_num(depot_feat)
        node_feat  = torch.nan_to_num(node_feat)

        B, N, _ = node_feat.shape

        x = torch.cat([
            self.embed_depot(depot_feat),   # (B, 1, D)
            self.embed_node(node_feat),     # (B, N, D)
        ], dim=1)  # (B, N+1, D)

        # Compute decay matrix once for all layers
        # Defensive reshape: ensure all tensors are (B, N+1, k) before cat
        def to_3d(t):
            """Ensure tensor is (B, N, k) — unsqueeze if 2D."""
            return t.unsqueeze(-1) if t.dim() == 2 else t

        dl = to_3d(td["demand_linehaul"])   # (B, N+1, 1)
        db = to_3d(td["demand_backhaul"])   # (B, N+1, 1)
        tw = to_3d(td["time_windows"])      # (B, N+1, 2)
        st = to_3d(td["service_time"])      # (B, N+1, 1)

        decay_feat = torch.cat([dl, db, tw, st], dim=-1)  # (B, N+1, 5)
        decay_feat = torch.nan_to_num(decay_feat)

        coords = td["locs"]  # (B, N+1, 2)
        D_mat  = self.decay_kernel(coords, decay_feat)  # (B, H, N+1, N+1)

        for layer in self.layers:
            x = layer(x, D=D_mat)

        return self.final_norm(x)  # (B, N+1, D)


# ============================================================================
# 4. DYNAMIC TASK MEMORY (DTM)
# ============================================================================

class DynamicTaskMemory(nn.Module):
    """
    Dynamic Task Memory — replaces static prompt conditioning.

    Static (CADA/MTPOMO):
        task_embed injected into encoder once → same conditioning every step

    Dynamic (ours):
        task_ctx_t = CrossAttn(query=E_task, key/value=S_global_t)
        q_t        = q_base_t + W_task · task_ctx_t

    Intuition:
        - E_task encodes WHICH constraints exist (e.g., has_TW, has_backhaul)
        - S_global encodes WHAT has happened so far in the tour
        - task_ctx = "given my tour history, which constraints are currently binding?"
        - This is DYNAMIC: late in a tour with heavy capacity used → capacity constraint
          becomes more relevant and S_global encodes this implicitly

    Implementation:
        E_task: (B, n_task_tokens, D) — from TaskEmbedding module
        S_global_flat: (B, S, H*D) — flattened global memory per start
        task_ctx: (B, S, D) — per-start dynamic task context
    """

    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: int,
                 n_constraints: int = 5):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim  = qkv_dim
        D = embedding_dim

        # Task embedding: binary constraint flags → D-dim per constraint token
        # 5 constraints: C (capacity), O (open), B (backhaul), L (limit), TW
        self.task_embed = nn.Embedding(n_constraints, D)
        nn.init.normal_(self.task_embed.weight, std=0.02)

        # Project S_global (H*D*D dimensional matrix → D per head, then pool) to K/V
        # We flatten S_global per head to qkv_dim^2, then project to qkv_dim
        S_flat_dim = head_num * qkv_dim  # after mean-pooling over key dim of S
        self.W_S_k = nn.Linear(S_flat_dim, head_num * qkv_dim, bias=False)
        self.W_S_v = nn.Linear(S_flat_dim, head_num * qkv_dim, bias=False)

        # Project task tokens to Q
        self.W_task_q = nn.Linear(D, head_num * qkv_dim, bias=False)

        # Project task context back to D
        self.W_task_out = nn.Linear(head_num * qkv_dim, D, bias=False)
        nn.init.zeros_(self.W_task_out.weight)  # init to zero → start as identity

        # Modulate decoder query
        self.W_task_mod = nn.Linear(D, head_num * qkv_dim, bias=False)
        nn.init.zeros_(self.W_task_mod.weight)  # conservative init

        self.norm = RMSNorm(D)
        self.scale = qkv_dim ** -0.5

    def get_task_embedding(self, td: TensorDict) -> Tensor:
        """
        Extract task constraint flags and produce task embeddings.
        td['p_s_tag']: (B, >=5) binary flags [C, O, B, L, TW]
        Returns: (B, 5, D) — one embedding per constraint type
        """
        # Use p_s_tag from td (same field CADA uses)
        flags = td['p_s_tag'][:, :5]  # (B, 5) binary
        idx = torch.arange(5, device=flags.device).unsqueeze(0)  # (1, 5)
        embs = self.task_embed(idx).expand(flags.shape[0], -1, -1)  # (B, 5, D)
        # Weight by flag: zero out inactive constraints
        embs = embs * flags.unsqueeze(-1)  # (B, 5, D)
        return embs  # (B, 5, D)

    def forward(
        self,
        S_global: Tensor,    # (B, num_starts, H, D_q, D_q) — global route memory
        task_emb: Tensor,    # (B, 5, D) — precomputed task embeddings
        q_base_flat: Tensor, # (BS, H*D) — base decoder queries in flat space
        num_starts: int,
    ) -> Tensor:
        """
        Returns task-modulated query addition: (BS, H*D)
        This gets ADDED to q_base_flat in the decoder.
        """
        B, S, H, D_q, _ = S_global.shape
        BS = B * S

        # --- S_global → K, V for cross-attention ---
        # S_global: (B, S, H, D_q, D_q)
        # Mean-pool over last dim to get a D_q vector per head:
        # (B, S, H, D_q) → flatten heads → (B, S, H*D_q)
        S_vec = S_global.mean(dim=-1)                    # (B, S, H, D_q)
        S_vec_flat = S_vec.reshape(B, S, H * D_q)        # (B, S, H*D_q)

        # Expand task_emb from (B, 5, D) to (B, S, 5, D)
        task_q = task_emb.unsqueeze(1).expand(B, S, 5, -1)  # (B, S, 5, D)

        # Flatten B*S for batch matmul
        S_vec_bs = S_vec_flat.reshape(BS, 1, H * D_q)       # (BS, 1, H*D_q)
        task_q_bs = task_q.reshape(BS, 5, -1)                # (BS, 5, D_emb)

        # Project S to K, V: (BS, H, 1, D_q)
        K = self.W_S_k(S_vec_bs).reshape(BS, 1, H, D_q).transpose(1, 2)  # (BS,H,1,D_q)
        V = self.W_S_v(S_vec_bs).reshape(BS, 1, H, D_q).transpose(1, 2)

        # Project task tokens to Q: (BS, H, 5, D_q)
        Q = self.W_task_q(task_q_bs).reshape(BS, 5, H, D_q).transpose(1, 2)

        # Cross-attention: task queries into memory K/V
        score = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (BS,H,5,1)
        weight = torch.softmax(score, dim=-1)
        ctx = torch.matmul(weight, V)                               # (BS,H,5,D_q)

        # Pool over task tokens (mean), project to D_emb
        ctx = ctx.mean(dim=2)                                       # (BS, H, D_q)
        ctx = ctx.reshape(BS, H * D_q)                              # (BS, H*D_q)
        task_ctx = self.W_task_out(ctx)                             # (BS, D_emb)
        task_ctx = self.norm(task_ctx)

        # Modulate decoder query
        delta_q = self.W_task_mod(task_ctx)                         # (BS, H*D_q)
        return delta_q  # added to q_base_flat in decoder


# ============================================================================
# 5. HIERARCHICAL ROUTE MEMORY (HRM)
# ============================================================================

class HierarchicalRouteMemory(nn.Module):
    """
    Two-level recurrent retention state.

    S_local[h]:  (B, S, H, D, D) — resets at depot, tracks current sub-tour
    S_global[h]: (B, S, H, D, D) — never resets, slow decay, cross-sub-tour patterns

    Update:
        kv = k ⊗ v                             # outer product
        S_local  ← 0            if at_depot    # hard reset
                 ← γ_l·S_local + kv   else
        S_global ← γ_g·S_global + kv           # always accumulate
                   + W_b·depot_emb             if at_depot  (boundary injection)

    Retrieval:
        y_l = q · S_local,   y_g = q · S_global
        gate = σ(W_gate([y_l | y_g | q]))
        y = gate⊙y_l + (1-gate)⊙y_g
    """

    def __init__(self, mp: dict):
        super().__init__()
        H  = mp['head_num']
        D  = mp['qkv_dim']
        De = mp['embedding_dim']
        self.H, self.D = H, D

        gl_init = mp.get('hrm_gamma_local_logit_init',  2.197)
        gg_init = mp.get('hrm_gamma_global_logit_init', 4.595)
        self.gamma_local  = nn.Parameter(torch.full((H,), gl_init))
        self.gamma_global = nn.Parameter(torch.full((H,), gg_init))

        b_std = mp.get('hrm_boundary_std', 0.01)
        self.W_boundary = nn.Linear(De, H * D * D, bias=False)
        nn.init.normal_(self.W_boundary.weight, std=b_std)

        self.W_gate = nn.Linear(H * D * 3, H * D, bias=True)
        nn.init.zeros_(self.W_gate.weight)
        nn.init.constant_(self.W_gate.bias, mp.get('hrm_gate_bias_init', 0.0))

        self.group_norm = nn.GroupNorm(H, H * D)
        self.W_out = nn.Linear(H * D, De)

    def init_states(self, B: int, S: int, device) -> Tuple[Tensor, Tensor]:
        shape = (B, S, self.H, self.D, self.D)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def forward(
        self,
        q: Tensor,           # (B, S, H, D)
        k: Tensor,           # (B, S, H, D)
        v: Tensor,           # (B, S, H, D)
        S_local: Tensor,     # (B, S, H, D, D)
        S_global: Tensor,    # (B, S, H, D, D)
        at_depot: Tensor,    # (B, S) bool
        depot_emb: Tensor,   # (B, 1, D_embed)
    ) -> Tuple[Tensor, Tensor, Tensor]:

        B, S, H, D = q.shape
        gl = torch.sigmoid(self.gamma_local ).view(1,1,H,1,1)
        gg = torch.sigmoid(self.gamma_global).view(1,1,H,1,1)

        kv  = torch.einsum('bshd,bshe->bshde', k, v)  # (B,S,H,D,D)
        dep = at_depot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,S,1,1,1)

        # S_local: hard reset at depot
        S_l_new = torch.where(dep, torch.zeros_like(S_local), gl * S_local + kv)

        # S_global: always accumulate + boundary injection at depot
        bsig = self.W_boundary(depot_emb.squeeze(1))      # (B, H*D*D)
        bsig = bsig.view(B, 1, H, D, D).expand(B, S, H, D, D)
        S_g_decay = gg * S_global + kv
        S_g_new = torch.where(dep, S_g_decay + bsig, S_g_decay)

        # Retrieve
        q_ = q.unsqueeze(-2)                               # (B,S,H,1,D)
        y_l = torch.matmul(q_, S_l_new).squeeze(-2)        # (B,S,H,D)
        y_g = torch.matmul(q_, S_g_new).squeeze(-2)

        yl_f = y_l.reshape(B, S, H*D)
        yg_f = y_g.reshape(B, S, H*D)
        q_f  = q.reshape(B, S, H*D)

        gate = torch.sigmoid(self.W_gate(torch.cat([yl_f, yg_f, q_f], -1)))
        y    = gate * yl_f + (1-gate) * yg_f               # (B,S,H*D)

        y_norm = self.group_norm(y.reshape(B*S, H*D, 1)).reshape(B, S, H*D)
        out = self.W_out(y_norm)                            # (B,S,D_embed)

        return out, S_l_new, S_g_new


# ============================================================================
# 6. VRP DECODER with HRM + DTM
# ============================================================================

class VRP_Decoder(nn.Module):
    """
    Per decoding step t:

    1. context  = [cur_node_emb | remaining_cap_l | remaining_cap_b |
                   current_time | route_length | open_route]
    2. q_base   = W_q(context)                      # (BS, H*D)
    3. delta_q  = DTM(S_global, task_emb)           # (BS, H*D) — dynamic task
    4. q        = q_base + delta_q                  # (BS, H, D)
    5. k, v     = W_k(cur_emb), W_v(cur_emb)
    6. HRM update: S_local', S_global' ← HRM(q,k,v,S_local,S_global)
    7. memory_out = HRM retrieval output            # (BS, D)
    8. cross_out  = CrossAttn(memory_out, encoder_nodes)  # (BS, D)
    9. logits   = pointer(cross_out)                # (BS, N+1)
    """

    def __init__(self, **mp):
        super().__init__()
        self.mp = mp
        D  = mp['embedding_dim']
        H  = mp['head_num']
        Dq = mp['qkv_dim']

        # Context → base query
        self.Wq_ctx = nn.Linear(D + 5, H * Dq, bias=False)

        # Current node → K, V for HRM update
        self.Wk_hrm = nn.Linear(D, H * Dq, bias=False)
        self.Wv_hrm = nn.Linear(D, H * Dq, bias=False)

        # HRM
        self.hrm = HierarchicalRouteMemory(mp)

        # DTM
        self.dtm = DynamicTaskMemory(D, H, Dq, n_constraints=5)

        # Cross-attention: memory_out → encoder nodes
        self.Wq_cross   = nn.Linear(D, H * Dq, bias=False)
        self.Wk_enc     = nn.Linear(D, H * Dq, bias=False)
        self.Wv_enc     = nn.Linear(D, H * Dq, bias=False)
        self.cross_proj = nn.Linear(H * Dq, D)

        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)

    def init_memory(self, B: int, S: int, device) -> Tuple[Tensor, Tensor]:
        return self.hrm.init_states(B, S, device)

    def forward(
        self,
        td,
        cache: PrecomputedCache,
        num_starts: int,
        S_local: Tensor,
        S_global: Tensor,
        task_emb: Tensor,       # (B, 5, D) — precomputed once before decode loop
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        td:       flat (BS, ...) where BS = B * num_starts
        Returns:  log_probs (BS, N+1), mask (BS, N+1), S_local, S_global
        """
        H  = self.mp['head_num']
        Dq = self.mp['qkv_dim']
        BS = td.batch_size[0]
        B  = BS // num_starts

        # --- Context ---
        cur_emb = gather_by_index(cache.node_embeddings, td["current_node"])  # (BS, D)
        rl    = td["vehicle_capacity"] - td["used_capacity_linehaul"]
        rb    = td["vehicle_capacity"] - td["used_capacity_backhaul"]
        state = torch.cat([rl, rb, td["current_time"],
                           td["current_route_length"], td["open_route"]], dim=-1)  # (BS,5)
        ctx   = torch.cat([cur_emb, state], dim=-1)   # (BS, D+5)

        # --- Base query ---
        q_base = self.Wq_ctx(ctx)                      # (BS, H*Dq)

        # --- Dynamic Task Memory modulation ---
        delta_q = self.dtm(S_global, task_emb, q_base, num_starts)  # (BS, H*Dq)
        q_mod   = (q_base + delta_q).reshape(B, num_starts, H, Dq)  # (B,S,H,Dq)

        # --- HRM K, V from current node ---
        k_hrm = self.Wk_hrm(cur_emb).reshape(B, num_starts, H, Dq)
        v_hrm = self.Wv_hrm(cur_emb).reshape(B, num_starts, H, Dq)

        # --- Depot detection ---
        at_depot  = (td["current_node"].squeeze(-1) == 0).view(B, num_starts)
        depot_emb = cache.node_embeddings[:, :1, :].view(B, num_starts, 1, -1)[:, 0, :, :]
        # (B, 1, D)

        # --- HRM forward ---
        mem_out, S_local, S_global = self.hrm(
            q_mod, k_hrm, v_hrm, S_local, S_global, at_depot, depot_emb
        )  # mem_out: (B, S, D)
        mem_flat = mem_out.view(BS, -1)               # (BS, D)

        # --- Cross-attention into encoder ---
        mask = td["action_mask"]                       # (BS, N+1)
        q_c  = self.Wq_cross(mem_flat).reshape(BS, H, 1, Dq)   # (BS,H,1,Dq)
        # cache.glimpse_key: (BS, H, N+1, Dq)
        sc   = torch.matmul(q_c, cache.glimpse_key.transpose(-2,-1)) * (Dq**-0.5)
        sc   = sc.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        w    = torch.softmax(sc, dim=-1)
        ca   = torch.matmul(w, cache.glimpse_val).squeeze(2).reshape(BS, H*Dq)
        ca   = self.cross_proj(ca)                    # (BS, D)

        out  = self.norm1(mem_flat + ca)              # (BS, D)

        # --- Pointer ---
        score = torch.matmul(out.unsqueeze(1), cache.logit_key).squeeze(1)
        score = score / self.mp['sqrt_embedding_dim']  # (BS, N+1)
        logits = torch.tanh(score) * self.mp['logit_clipping']
        logits[~mask] = float('-inf')

        return F.log_softmax(logits, dim=-1), mask, S_local, S_global


# ============================================================================
# 7. MAIN MODEL
# ============================================================================

class VRPModel(nn.Module):
    """
    HRM-NCO: Hierarchical Route Memory for Neural Combinatorial Optimization.

    Contributions:
      (1) Spatial-Semantic Decay Kernel in encoder
      (2) Hierarchical Route Memory in decoder
      (3) Dynamic Task Memory for multi-task conditioning

    No dependency on CADA architecture. Built on pure RetNet backbone.
    """

    def __init__(self, args):
        super().__init__()
        self.args   = args
        self.mp     = args.model_params
        self.encoder = VRP_Encoder(**args.model_params)
        self.decoder = VRP_Decoder(**args.model_params)

    @staticmethod
    def greedy(logprobs):
        return logprobs.argmax(dim=-1)

    @staticmethod
    def sampling(logprobs, log, mask):
        probs    = logprobs.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)
        while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
            log("Sampled bad values, resampling!")
            selected = torch.multinomial(probs, 1).squeeze(1)
        return selected

    def forward(self, td, env):
        mp  = self.mp
        H   = mp['head_num']
        Dq  = mp['qkv_dim']

        # ---- Encode ----
        node_embed = self.encoder(td)   # (B, N+1, D)
        B_orig     = node_embed.size(0)

        # ---- Precompute task embeddings (once, static per instance) ----
        # DTM will use these dynamically at each step
        task_emb = self.decoder.dtm.get_task_embedding(td)  # (B, 5, D)

        # ---- Multi-start setup ----
        num_starts, action = env.select_start_nodes(td)
        td = batchify(td, num_starts)
        logprobs_list = [torch.zeros_like(action, device=td.device)]
        actions_list  = [action]
        td.set("action", action)
        td = env.step(td)["next"]

        # ---- Precompute encoder K/V, batchify for multi-start ----
        enc_k  = node_embed.reshape(B_orig, -1, H, Dq
                    ).transpose(1,2).contiguous()
        # Actually use proper linear projections:
        enc_k  = self.decoder.Wk_enc(node_embed)  # (B, N+1, H*Dq)
        enc_k  = enc_k.reshape(B_orig, -1, H, Dq).transpose(1,2)  # (B,H,N+1,Dq)
        enc_v  = self.decoder.Wv_enc(node_embed).reshape(B_orig, -1, H, Dq).transpose(1,2)
        logit_k = node_embed.transpose(1, 2)       # (B, D, N+1)

        cache = PrecomputedCache(
            batchify(node_embed, num_starts),
            batchify(enc_k,      num_starts),
            batchify(enc_v,      num_starts),
            batchify(logit_k,    num_starts),
        )

        # Batchify task_emb for flat (BS, ...) decoder
        task_emb_bs = batchify(task_emb, num_starts)  # (BS, 5, D)
        # But DTM needs (B, 5, D) for S_global indexing — pass both
        # We pass flat task_emb_bs; DTM will reshape using num_starts

        # ---- Initialize HRM states ----
        S_local, S_global = self.decoder.init_memory(B_orig, num_starts, node_embed.device)

        # ---- Decode loop ----
        while not td["done"].all():
            logprobs, mask, S_local, S_global = self.decoder(
                td, cache, num_starts, S_local, S_global, task_emb_bs
            )
            if self.training:
                select = VRPModel.sampling(logprobs, self.args.log, mask)
            else:
                select = VRPModel.greedy(logprobs)

            logprobs = gather_by_index(logprobs, select, dim=1)
            td.set("action", select)
            actions_list.append(select)
            logprobs_list.append(logprobs)
            td = env.step(td)["next"]

        # ---- Post-process ----
        logprobs = torch.stack(logprobs_list, 1)
        actions  = torch.stack(actions_list, 1)
        td.set("reward", env.get_reward(td, actions))
        assert (logprobs > -1000).data.all(), "Log-probs contain -inf!"
        return {
            "reward":         td["reward"],
            "log_likelihood": logprobs.sum(1),
        }