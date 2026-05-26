import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from utils.functions import batchify, gather_by_index, unbatchify


def linear_layer(input_dim, output_dim, std=1e-2, bias=True):
    layer = nn.Linear(input_dim, output_dim, bias=bias)
    nn.init.normal_(layer.weight, std=std)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


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
        return PrecomputedCache(
            *[
                batchify(emb, num_starts)
                if isinstance(emb, Tensor) or isinstance(emb, TensorDict)
                else emb
                for emb in self.fields
            ]
        )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class ConstraintStateEncoder(nn.Module):
    """Encodes active VRP constraints as a dynamic state, not as CaDA prompt tokens."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, embedding_dim),
            RMSNorm(embedding_dim),
            nn.GELU(),
            linear_layer(embedding_dim, embedding_dim),
        )

    def forward(self, td):
        p_s_tag = td["p_s_tag"]
        if p_s_tag.size(-1) == 5:
            size_tag = torch.full_like(p_s_tag[:, :1], (td["locs"].shape[1] - 1) / 2000)
            p_s_tag = torch.cat((p_s_tag, size_tag), dim=-1)
        return self.net(p_s_tag[:, :6])


class GeoPositionEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, embedding_dim),
            RMSNorm(embedding_dim),
            nn.GELU(),
            linear_layer(embedding_dim, embedding_dim),
        )

    def forward(self, locs):
        depot = locs[:, :1, :]
        rel = locs - depot
        dist = rel.norm(p=2, dim=-1, keepdim=True)
        geo = torch.cat((locs, dist), dim=-1)
        return self.proj(geo)


class Retention(nn.Module):
    """Parallel retention over node order with distance decay and no softmax."""

    def __init__(self, embedding_dim, head_num, qkv_dim, gamma=0.95):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.gamma = gamma
        self.wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.gate = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.out = nn.Linear(head_num * qkv_dim, embedding_dim)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        q = reshape_by_heads(self.wq(x), self.head_num)
        k = reshape_by_heads(self.wk(x), self.head_num)
        v = reshape_by_heads(self.wv(x), self.head_num)
        gate = F.silu(self.gate(x))

        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.qkv_dim)
        idx = torch.arange(seq_len, device=x.device)
        decay = self.gamma ** (idx[:, None] - idx[None, :]).abs().float()
        score = score * decay.view(1, 1, seq_len, seq_len)

        out = torch.matmul(score, v) / max(seq_len, 1)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.head_num * self.qkv_dim)
        return self.out(out * gate)


class RetentionBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, ff_hidden_dim, gamma):
        super().__init__()
        self.retention = Retention(embedding_dim, head_num, qkv_dim, gamma=gamma)
        self.norm1 = RMSNorm(embedding_dim)
        self.ff = SwiGLU(embedding_dim, ff_hidden_dim)
        self.norm2 = RMSNorm(embedding_dim)

    def forward(self, x):
        x = self.norm1(x + self.retention(x))
        x = self.norm2(x + self.ff(x))
        return x


class RouteStateMemory(nn.Module):
    """Two-level retentive memory: route-local memory and instance-level memory."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(embedding_dim + 5, embedding_dim),
            RMSNorm(embedding_dim),
            nn.GELU(),
        )
        self.route_decay = nn.Parameter(torch.tensor(0.90))
        self.instance_decay = nn.Parameter(torch.tensor(0.97))

    def forward(self, token_embedding, scalar_state, route_memory, instance_memory, reset_route):
        token = self.token_proj(torch.cat((token_embedding, scalar_state), dim=-1))
        route_gamma = self.route_decay.sigmoid()
        instance_gamma = self.instance_decay.sigmoid()
        route_memory = route_gamma * route_memory + token
        instance_memory = instance_gamma * instance_memory + token
        route_memory = torch.where(reset_route[:, None], torch.zeros_like(route_memory), route_memory)
        return route_memory, instance_memory


class VRPModel(nn.Module):
    """Retentive route-state VRP solver.

    This model intentionally keeps only the CaDA repository interface. Its architecture is
    a route-construction memory model, not a dual-attention prompt model.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = RetentiveVRPEncoder(**args.model_params)
        self.decoder = RetentiveVRPDecoder(**args.model_params)

    @staticmethod
    def greedy(logprobs, mask=None):
        selected = logprobs.argmax(dim=-1)
        if mask is not None:
            assert not (~mask).gather(1, selected.unsqueeze(-1)).data.any(), "infeasible action selected"
        return selected

    @staticmethod
    def sampling(logprobs, log, mask=None):
        probs = logprobs.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)
        if mask is not None:
            while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
                log("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
            assert not (~mask).gather(1, selected.unsqueeze(-1)).data.any(), "infeasible action selected"
        return selected

    def forward(self, td, env):
        args = self.args
        node_embed = self.encoder(td)
        num_starts, action = env.select_start_nodes(td)

        td = batchify(td, num_starts)
        node_embed_batched = batchify(node_embed, num_starts)
        route_memory = torch.zeros(td.batch_size[0], node_embed.size(-1), device=td.device)
        instance_memory = torch.zeros_like(route_memory)

        logprobs_list = [torch.zeros_like(action, device=td.device)]
        actions_list = [action]
        td.set("action", action)
        selected_embed = gather_by_index(node_embed_batched, action)
        scalar_state = self.decoder.scalar_state(td)
        route_memory, instance_memory = self.decoder.memory(
            selected_embed,
            scalar_state,
            route_memory,
            instance_memory,
            reset_route=(action == 0),
        )
        td = env.step(td)["next"]

        decoder_k = reshape_by_heads(self.decoder.wk(node_embed), head_num=args.model_params["head_num"])
        decoder_v = reshape_by_heads(self.decoder.wv(node_embed), head_num=args.model_params["head_num"])
        decoder_single_head_k = node_embed.transpose(1, 2)
        cache = PrecomputedCache(node_embed, decoder_k, decoder_v, decoder_single_head_k)

        while not td["done"].all():
            logprobs, mask = self.decoder(td, cache, num_starts, route_memory, instance_memory)
            if self.training:
                selected = VRPModel.sampling(logprobs, args.log, mask)
            else:
                selected = VRPModel.greedy(logprobs, mask)
            logprobs = gather_by_index(logprobs, selected, dim=1)
            td.set("action", selected)
            actions_list.append(selected)
            logprobs_list.append(logprobs)

            selected_embed = gather_by_index(node_embed_batched, selected)
            scalar_state = self.decoder.scalar_state(td)
            route_memory, instance_memory = self.decoder.memory(
                selected_embed,
                scalar_state,
                route_memory,
                instance_memory,
                reset_route=(selected == 0),
            )
            td = env.step(td)["next"]

        logprobs = torch.stack(logprobs_list, 1)
        actions = torch.stack(actions_list, 1)
        td.set("reward", env.get_reward(td, actions))
        assert (logprobs > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        return {"reward": td["reward"], "log_likelihood": logprobs.sum(1)}


class RetentiveVRPEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.embedding_depot = nn.Linear(3, embedding_dim)
        self.embedding_node = nn.Linear(7, embedding_dim)
        self.constraint_encoder = ConstraintStateEncoder(embedding_dim)
        self.position = GeoPositionEncoding(embedding_dim)
        layer_num = model_params["encoder_layer_num"]
        gamma = model_params.get("retention_gamma", 0.95)
        self.layers = nn.ModuleList(
            [
                RetentionBlock(
                    embedding_dim,
                    model_params["head_num"],
                    model_params["qkv_dim"],
                    model_params["ff_hidden_dim"],
                    gamma,
                )
                for _ in range(layer_num)
            ]
        )

    def forward(self, td):
        depot_feats = torch.cat((td["locs"][:, :1, :], td["distance_limit"][..., None]), -1)
        node_feats = torch.cat(
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["locs"][:, 1:, :],
            ),
            -1,
        )
        depot_feats = torch.nan_to_num(depot_feats, nan=0.0, posinf=0.0, neginf=0.0)
        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.cat((self.embedding_depot(depot_feats), self.embedding_node(node_feats)), -2)

        constraint_state = self.constraint_encoder(td).unsqueeze(1)
        x = x + constraint_state
        x = x + self.position(td["locs"])
        for layer in self.layers:
            x = layer(x)
        return x


class RetentiveVRPDecoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]
        self.state_proj = nn.Sequential(nn.Linear(5, embedding_dim), RMSNorm(embedding_dim), nn.GELU())
        self.wq_last = nn.Linear(embedding_dim * 4, head_num * qkv_dim, bias=False)
        self.wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.memory = RouteStateMemory(embedding_dim)

    @staticmethod
    def scalar_state(td):
        return torch.cat(
            [
                td["vehicle_capacity"] - td["used_capacity_linehaul"],
                td["vehicle_capacity"] - td["used_capacity_backhaul"],
                td["current_time"],
                td["current_route_length"],
                td["open_route"].float(),
            ],
            -1,
        )

    def forward(self, td, cache, num_starts, route_memory, instance_memory):
        td = unbatchify(td, num_starts)
        route_memory = unbatchify(route_memory, num_starts)
        instance_memory = unbatchify(instance_memory, num_starts)

        cur_node_embedding = gather_by_index(cache.node_embeddings, td["current_node"])
        state_embedding = self.state_proj(self.scalar_state(td))
        context_embedding = torch.cat(
            (cur_node_embedding, state_embedding, route_memory, instance_memory), dim=-1
        )
        glimpse_q = reshape_by_heads(self.wq_last(context_embedding), self.model_params["head_num"])
        mask = td["action_mask"]
        out_concat = multi_head_attention(glimpse_q, cache.glimpse_key, cache.glimpse_val, mask)
        mh_atten_out = self.multi_head_combine(out_concat)
        score = torch.matmul(mh_atten_out, cache.logit_key)
        score_scaled = score / self.model_params["sqrt_embedding_dim"]

        logits = rearrange(score_scaled, "b s l -> (s b) l", s=num_starts)
        mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        logits = torch.tanh(logits) * self.model_params["logit_clipping"]
        logits[~mask] = float("-inf")
        return F.log_softmax(logits, dim=-1), mask


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    return qkv.reshape(batch_s, n, head_num, -1).transpose(1, 2)


def multi_head_attention(q, k, v, ninf_mask=None):
    batch_s, head_num, n, key_dim = q.shape
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(key_dim)
    if ninf_mask is not None:
        additive_mask = torch.zeros_like(score)
        additive_mask = additive_mask.masked_fill(~ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s), float("-inf"))
        score = score + additive_mask
    weights = F.softmax(score, dim=3)
    out = torch.matmul(weights, v)
    return out.transpose(1, 2).contiguous().view(batch_s, n, head_num * key_dim)
