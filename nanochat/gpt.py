"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
- Parallel interleaved training: recurrent h_final feedback, trained as 1 full pass
  + N partial (interleaved) passes that recompute an interleaved subset of positions
  with fresh feedback

Parallel interleaved training in one paragraph: the model runs a normal full forward first
(pass 0), capturing every layer's K,V into a buffer. The final hidden state
h_final, shifted right by one position (token t sees t-1, preserving causality),
is fed back into the network three ways at every layer: added to the residual
stream (gamma injection), projected into V space, and projected into K space,
where a learned triple gate mixes the feedback V with the local V and a
token-indexed init value embedding. Each interleaved pass i then recomputes only the
positions with pos % N == i against the shared KV buffer, writes its fresh K,V
back, and merges its h_final into the running feedback — approximating
token-level recurrence at a fraction of the cost of full passes. Optionally
(interleaved_kv_refresh) the buffered V/K at not-yet-visited positions are cheaply
re-projected from the latest h_final between passes to reduce staleness.
At inference the model decodes token-by-token with true recurrence (see
loss_eval.evaluate_bpb_sequential). The loss is the mean over all passes.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Feature toggles for ablation studies
    use_value_embeds: bool = True  # Value embeddings (token-indexed); unused when use_interleaved (feedback replaces them)
    ve_every_layer: bool = False   # Value embeddings on EVERY layer (default: alternating, last layer included)
    use_x0_residual: bool = True   # Skip connection to input embedding
    use_resid_lambdas: bool = True # Learnable per-layer residual scaling
    ve_gate_channels: int = 32     # Number of input channels for value/feedback gate (0 = full n_embd)
    # ---- Parallel interleaved training (recurrent feedback) ----
    # Fixed recipe, applied at EVERY layer: h_final feedback is injected into the residual
    # stream (gamma), projected to V and K space (per-layer projections), and mixed via a
    # triple gate v = gate_local*v + gate_fb*v_fb + gate_ve*v_ve where v_ve is a
    # token-indexed init value embedding. Interleaved pass i covers positions pos % N == i.
    use_interleaved: bool = False       # Enable parallel interleaved training: 1 full pass + interleaved_num_passes interleaved passes
    interleaved_num_passes: int = 2     # Number of interleaved passes after the single full pass
    interleaved_kv_refresh: bool = False  # Between interleaved passes, cheaply refresh the buffered V/K at not-yet-visited positions by re-projecting the latest h_final. Reduces stale-KV gap.


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer, use_value_embeds=True, every_layer=False):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included;
    every_layer=True puts one on every layer)."""
    if not use_value_embeds:
        return False
    if every_layer:
        return True
    return layer_idx % 2 == (n_layer - 1) % 2


class FeedbackProjection(nn.Module):
    """Project h_final (last hidden states) into value (or key) space for the
    recurrent feedback channel: v_fb = W_v(h_final), reshaped to KV heads."""

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.W_v = nn.Linear(self.n_embd, self.kv_dim, bias=False)

    def forward(self, h_final):
        """
        Args:
            h_final: [B, T, n_embd] - final hidden states (after final norm)
        Returns:
            v_fb: [B, T, n_kv_head, head_dim] - projected values
        """
        B, T, _ = h_final.shape
        return self.W_v(h_final).view(B, T, self.n_kv_head, self.head_dim)


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = config.ve_gate_channels if config.ve_gate_channels > 0 else config.n_embd
        # Triple gate (interleaved models, every layer): v = gate_local * v + gate_fb * v_fb + gate_ve * v_ve
        self.triple_gate = nn.Linear(self.ve_gate_channels, 3 * self.n_kv_head, bias=False) if config.use_interleaved else None
        # Shared gate for token-indexed value embeddings (non-interleaved models)
        has_ve_gate = has_ve(layer_idx, config.n_layer, config.use_value_embeds, config.ve_every_layer)
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if (
            has_ve_gate and not config.use_interleaved
        ) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, v_fb=None, v_ve=None, k_fb=None, pass_idx=0):
        B, T, _ = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual: feedback triple gate (interleaved) OR token-indexed value embedding
        if self.triple_gate is not None and (v_fb is not None or v_ve is not None):
            # Triple gate: v = gate_local * v + gate_fb * v_fb + gate_ve * v_ve
            # v_fb is None only when the caller supplies no feedback (plain forward);
            # that is equivalent to pass 0 with zero feedback, so the term is dropped.
            gate_all = 2 * torch.sigmoid(self.triple_gate(x[..., :self.ve_gate_channels]))  # (B, T, 3*n_kv_head)
            gate_local, gate_fb, gate_ve = gate_all.split(self.n_kv_head, dim=-1)  # each (B, T, n_kv_head)
            gate = gate_fb  # for diagnostics (gate_fb)
            self._last_gate_local = gate_local  # for diagnostics (gate_local)
            self._last_gate_ve = gate_ve  # for diagnostics (gate_ve)
            v = gate_local.unsqueeze(-1) * v
            if v_fb is not None:
                v = v + gate_fb.unsqueeze(-1) * v_fb
            if v_ve is not None:
                v = v + gate_ve.unsqueeze(-1) * v_ve
            if k_fb is not None:
                k = gate_local.unsqueeze(-1) * k + gate_fb.unsqueeze(-1) * k_fb
        elif ve is not None:
            # Token embedding style: use per-token value embeddings
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve
        else:
            gate = None
        # Record gate stats if diagnostics are enabled (keyed by pass_idx)
        if gate is not None and getattr(self, '_collect_gate_stats', False):
            p = getattr(self, '_current_pass_idx', 0)
            if p not in self._gate_sums:
                self._gate_sums[p] = 0.0
                self._gate_counts[p] = 0
            self._gate_sums[p] += gate.detach().mean().item()
            self._gate_counts[p] += 1
            # Track gate_local for triple-gate mode
            if hasattr(self, '_last_gate_local') and self._last_gate_local is not None:
                if not hasattr(self, '_gate_local_sums'):
                    self._gate_local_sums = {}
                if p not in self._gate_local_sums:
                    self._gate_local_sums[p] = 0.0
                self._gate_local_sums[p] += self._last_gate_local.detach().mean().item()
                self._last_gate_local = None
            # Track gate_ve for triple-gate mode
            if hasattr(self, '_last_gate_ve') and self._last_gate_ve is not None:
                if not hasattr(self, '_gate_ve_sums'):
                    self._gate_ve_sums = {}
                if p not in self._gate_ve_sums:
                    self._gate_ve_sums[p] = 0.0
                self._gate_ve_sums[p] += self._last_gate_ve.detach().mean().item()
                self._last_gate_ve = None

        # Capture K pre-rotary, pre-norm (for cheap K refresh after h_final updates).
        # At this point k = gate_local*k_local + gate_fb*k_fb — linear in the feedback
        # contribution, so the refresh can do:
        # k_prerot_new = k_prerot + gate_fb*(k_fb_new - k_fb_old), then re-apply
        # rotary + RMSNorm. Captured only when both _capture_kv and _capture_gate_fb
        # are set (the refresh feature's combined flag).
        if getattr(self, '_capture_kv', False) and getattr(self, '_capture_gate_fb', False):
            self._captured_k_prerot = k

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Capture K,V for the interleaved KV buffer (after rotary + QK norm for K, after value residual for V)
        # Not detached so gradients flow from interleaved passes back through the KV buffer
        if getattr(self, '_capture_kv', False):
            self._captured_k = k
            self._captured_v = v
            # Also capture the gate that multiplies v_fb (for cheap V buffer refresh after
            # interleaved passes).
            if getattr(self, '_capture_gate_fb', False):
                self._captured_gate_fb = locals().get('gate_fb', None)

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def forward_interleaved(self, x, ve, cos_sin, window_size, kv_k, kv_v, interleaved_positions,
                       v_fb=None, v_ve=None, k_fb=None, pass_idx=0):
        """Interleaved attention: compute fresh Q,K,V at interleaved positions, write K,V back to buffer, attend to updated buffer."""
        B, T_interleaved, _ = x.size()

        # Compute Q, K, V projections at interleaved positions
        q = self.c_q(x).view(B, T_interleaved, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T_interleaved, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T_interleaved, self.n_kv_head, self.head_dim)

        # Value residual (same triple-gate logic as normal forward, at interleaved positions)
        gate_all = 2 * torch.sigmoid(self.triple_gate(x[..., :self.ve_gate_channels]))
        gate_local, gate_fb, gate_ve = gate_all.split(self.n_kv_head, dim=-1)
        v = gate_local.unsqueeze(-1) * v + gate_fb.unsqueeze(-1) * v_fb + gate_ve.unsqueeze(-1) * v_ve
        k = gate_local.unsqueeze(-1) * k + gate_fb.unsqueeze(-1) * k_fb

        # Rotary + QK norm
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # For attention: use fresh K,V at ALL interleaved positions (best context for this pass)
        # Match buffer dtype: _refresh_kv_buffer can promote the buffer to fp32 (norm fallback
        # under autocast), while fresh k/v here may still be bf16. Cast to avoid index_put dtype mismatch.
        kv_k_attn = kv_k.clone()
        kv_k_attn[:, interleaved_positions] = k.to(kv_k_attn.dtype)
        kv_v_attn = kv_v.clone()
        kv_v_attn[:, interleaved_positions] = v.to(kv_v_attn.dtype)

        # Interleaved attention: Q at interleaved_positions, K/V from updated full buffer
        y = flash_attn.flash_attn_interleaved(q, kv_k_attn, kv_v_attn, interleaved_positions, window_size)

        y = y.contiguous().view(B, T_interleaved, -1)
        y = self.c_proj(y)

        return y, kv_k_attn, kv_v_attn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, v_fb=None, v_ve=None, k_fb=None, pass_idx=0):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, v_fb=v_fb, v_ve=v_ve, k_fb=k_fb, pass_idx=pass_idx)
        x = x + self.mlp(norm(x))
        return x

    def forward_interleaved(self, x, ve, cos_sin, window_size, kv_k, kv_v, interleaved_positions,
                       v_fb=None, v_ve=None, k_fb=None, pass_idx=0):
        """Block forward for interleaved passes: compute fresh K,V, write back to buffer, attend to updated buffer."""
        attn_out, kv_k, kv_v = self.attn.forward_interleaved(norm(x), ve, cos_sin, window_size, kv_k, kv_v, interleaved_positions, v_fb=v_fb, v_ve=v_ve, k_fb=k_fb, pass_idx=pass_idx)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, kv_k, kv_v


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        if config.use_resid_lambdas:
            self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        if config.use_x0_residual:
            self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings: alternating layers, last layer always included.
        # Not created when use_interleaved (the feedback triple gate replaces them at every layer).
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({} if config.use_interleaved else {str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer, config.use_value_embeds, config.ve_every_layer)})
        # Recurrent feedback mechanism (interleaved training) — one projection set per layer
        if config.use_interleaved:
            self.interleaved_projections = nn.ModuleDict({
                str(i): FeedbackProjection(config) for i in range(config.n_layer)
            })
            self.interleaved_key_projections = nn.ModuleDict({
                str(i): FeedbackProjection(config) for i in range(config.n_layer)
            })
            # Hidden feedback injection: x = α*x + β*x0 + γ*h_final_prev (γ = interleaved_lambdas)
            self.interleaved_lambdas = nn.Parameter(torch.zeros(config.n_layer))
            # Token-indexed init value embeddings: the v_ve channel of the triple gate
            self.interleaved_init_value_embeds = nn.ModuleDict({
                str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer)
            })
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        if self.config.use_resid_lambdas:
            self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        if self.config.use_x0_residual:
            self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Feedback (interleaved) initialization
        if self.config.use_interleaved:
            # Projections: init like c_v (uniform with same std)
            for proj in self.interleaved_projections.values():
                torch.nn.init.uniform_(proj.W_v.weight, -s, s)
            for proj in self.interleaved_key_projections.values():
                torch.nn.init.uniform_(proj.W_v.weight, -s, s)
            # Hidden injection: init interleaved_lambdas to small value (like x0_lambdas)
            self.interleaved_lambdas.fill_(0.1)
            # Init value embeds
            for ve in self.interleaved_init_value_embeds.values():
                torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.triple_gate is not None:
                torch.nn.init.zeros_(block.attn.triple_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)
            if self.config.use_interleaved:
                for ve in self.interleaved_init_value_embeds.values():
                    ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # interleaved the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # interleaved the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        interleaved_scalar_numel = 0
        interleaved_init_embeds_numel = 0
        if self.config.use_interleaved:
            interleaved_scalar_numel += self.interleaved_lambdas.numel()
            interleaved_init_embeds_numel = sum(p.numel() for p in self.interleaved_init_value_embeds.parameters())
        scalars_numel = (getattr(self, 'resid_lambdas', torch.empty(0)).numel() +
                         getattr(self, 'x0_lambdas', torch.empty(0)).numel() +
                         interleaved_scalar_numel)
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                           interleaved_init_embeds_numel + scalars_numel)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        # Feedback (interleaved) params
        interleaved_matrices = 0
        interleaved_scalars = 0
        interleaved_init_embeds = 0
        if self.config.use_interleaved:
            for proj in self.interleaved_projections.values():
                interleaved_matrices += proj.W_v.weight.numel()
            for proj in self.interleaved_key_projections.values():
                interleaved_matrices += proj.W_v.weight.numel()
            interleaved_scalars += self.interleaved_lambdas.numel()
            interleaved_init_embeds = sum(p.numel() for p in self.interleaved_init_value_embeds.parameters())
        scalars = (getattr(self, 'resid_lambdas', torch.empty(0)).numel() +
                   getattr(self, 'x0_lambdas', torch.empty(0)).numel() +
                   interleaved_scalars)
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + interleaved_matrices + interleaved_init_embeds
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'interleaved_matrices': interleaved_matrices,
            'interleaved_init_embeds': interleaved_init_embeds,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas] if hasattr(self, 'resid_lambdas') else []
        x0_params = [self.x0_lambdas] if hasattr(self, 'x0_lambdas') else []
        # Feedback (interleaved) params
        interleaved_matrix_params = []
        interleaved_scalar_params = []
        interleaved_init_embed_params = []
        if self.config.use_interleaved:
            for proj in self.interleaved_projections.values():
                interleaved_matrix_params.append(proj.W_v.weight)
            for proj in self.interleaved_key_projections.values():
                interleaved_matrix_params.append(proj.W_v.weight)
            interleaved_scalar_params.append(self.interleaved_lambdas)
            interleaved_init_embed_params = list(self.interleaved_init_value_embeds.parameters())
        all_grouped = (len(matrix_params) + len(embedding_params) + len(lm_head_params) +
                       len(value_embeds_params) + len(resid_params) + len(x0_params) +
                       len(interleaved_matrix_params) + len(interleaved_scalar_params) +
                       len(interleaved_init_embed_params))
        assert len(list(self.parameters())) == all_grouped, f"Parameter count mismatch: {len(list(self.parameters()))} != {all_grouped}"

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        if resid_params:
            param_groups.append(dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        if x0_params:
            param_groups.append(dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0))
        if value_embeds_params:
            param_groups.append(dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        if interleaved_scalar_params:
            param_groups.append(dict(kind='adamw', params=interleaved_scalar_params, lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        if interleaved_init_embed_params:
            param_groups.append(dict(kind='adamw', params=interleaved_init_embed_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        all_matrix_params = matrix_params + interleaved_matrix_params
        for shape in sorted({p.shape for p in all_matrix_params}):
            group_params = [p for p in all_matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', h_final_prev=None, pass_idx=0, return_hidden=False):
        """
        Forward pass with optional recurrent feedback (interleaved training / recurrent inference).

        Args:
            idx: Input token ids [B, T]
            targets: Target token ids for loss computation [B, T]
            kv_cache: KV cache for inference
            loss_reduction: 'mean' or 'none' for loss reduction
            h_final_prev: Feedback hidden states (shifted right) [B, T, n_embd].
                          None on a interleaved model = pass-0 semantics (zero feedback).
            pass_idx: Current pass index (for gate diagnostics only)
            return_hidden: If True, return (logits, h_final) even when targets is None

        Returns:
            If targets is not None: (loss, h_final) where h_final is the next-pass feedback
            If targets is None and return_hidden: (logits, h_final)
            If targets is None: logits
        """
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token
        x = norm(x)

        x0 = x if self.config.use_x0_residual else None
        use_interleaved = self.config.use_interleaved
        head_dim = self.config.n_embd // self.config.n_head
        for i, block in enumerate(self.transformer.h):
            # Propagate pass_idx to attention for per-pass gate diagnostics
            if getattr(block.attn, '_collect_gate_stats', False):
                block.attn._current_pass_idx = pass_idx
            if self.config.use_resid_lambdas:
                x = self.resid_lambdas[i] * x
            if self.config.use_x0_residual:
                x = x + self.x0_lambdas[i] * x0
            if use_interleaved:
                # Hidden feedback injection: add h_final_prev to x before each block
                if h_final_prev is not None:
                    x = x + self.interleaved_lambdas[i] * h_final_prev
                ve = None
                layer_v_fb = self.interleaved_projections[str(i)](h_final_prev) if h_final_prev is not None else None
                layer_k_fb = self.interleaved_key_projections[str(i)](h_final_prev) if h_final_prev is not None else None
                layer_v_ve = self.interleaved_init_value_embeds[str(i)](idx).view(B, T, self.config.n_kv_head, head_dim)
            else:
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                layer_v_fb = None
                layer_k_fb = None
                layer_v_ve = None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, v_fb=layer_v_fb, v_ve=layer_v_ve, k_fb=layer_k_fb, pass_idx=pass_idx)

        # h_final: final-block output, post-norm. Used for next-token logits AND as the
        # recurrent feedback returned for the next pass.
        h_final = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(h_final) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss, h_final
        else:
            # inference: return logits, optionally with hidden states for recurrent eval
            if return_hidden:
                return logits, h_final
            return logits

    def _shift_right(self, h):
        """
        Shift hidden states right by 1 position for feedback.
        h_shifted[t] = h[t-1], h_shifted[0] = zeros
        """
        zeros = torch.zeros_like(h[:, :1, :])
        return torch.cat([zeros, h[:, :-1, :]], dim=1)

    def _make_zero_h_final(self, idx):
        """Create zero tensor for h_final_prev on the full pass (for torch.compile compatibility)."""
        B, T = idx.size()
        return torch.zeros(B, T, self.config.n_embd, device=idx.device, dtype=self.transformer.wte.weight.dtype)

    def _forward_with_kv_capture(self, *args, capture_gate_fb=False, **kwargs):
        """Run forward and capture per-layer K,V (after rotary+QK norm for K, after value residual for V).
        If capture_gate_fb=True, also returns (gate_buffer, k_prerot_buffer):
          gate_buffer[i]     = gate_fb captured at layer i
          k_prerot_buffer[i] = pre-rotary, pre-norm K captured at layer i
        """
        for block in self.transformer.h:
            block.attn._capture_kv = True
            if capture_gate_fb:
                block.attn._capture_gate_fb = True
        result = self.forward(*args, **kwargs)
        kv_buffer = []
        gate_buffer = []
        k_prerot_buffer = []
        for block in self.transformer.h:
            kv_buffer.append((block.attn._captured_k, block.attn._captured_v))
            block.attn._capture_kv = False
            del block.attn._captured_k, block.attn._captured_v
            if capture_gate_fb:
                gate_buffer.append(getattr(block.attn, '_captured_gate_fb', None))
                k_prerot_buffer.append(getattr(block.attn, '_captured_k_prerot', None))
                block.attn._capture_gate_fb = False
                if hasattr(block.attn, '_captured_gate_fb'):
                    del block.attn._captured_gate_fb
                if hasattr(block.attn, '_captured_k_prerot'):
                    del block.attn._captured_k_prerot
        if capture_gate_fb:
            return result, kv_buffer, gate_buffer, k_prerot_buffer
        return result, kv_buffer

    def _refresh_kv_buffer(self, kv_buffer, gate_buffer, k_prerot_buffer,
                           v_fb_old, k_fb_old, h_final, visited_mask):
        """
        Cheaply refresh BOTH V and K in the buffer at NEVER-visited positions.

        V refresh (linear, easy): captured V = baseline + gate_fb * v_fb_old. Adds
        gate_fb * (v_fb_new - v_fb_old) at never-visited positions.

        K refresh (needs rotary + RMSNorm recompute): captured K is post-rotary + post-norm
        (nonlinear), so we also captured the PRE-rotary, PRE-norm K:
          k_prerot = gate_local * k_local + gate_fb * k_fb_old
        Refresh:
          k_prerot_new = k_prerot + gate_fb * (k_fb_new - k_fb_old)   at never-visited
          k_new        = norm(apply_rotary_emb(k_prerot_new, cos, sin))
        At visited positions, keep the buffer's current K (forward_interleaved wrote fresh K there).

        Args:
            kv_buffer: list of (k, v) per layer. Updated functionally (clone-then-combine).
            gate_buffer: list of gate_fb per layer, [B, T, n_kv_head].
            k_prerot_buffer: list of pre-rotary pre-norm K per layer. Mutated in-place.
            v_fb_old: list of [B, T, n_kv_head, head_dim] (or 0 sentinel). Updated in-place.
            k_fb_old: list of [B, T, n_kv_head, head_dim] (or 0 sentinel). Updated in-place.
            h_final: [B, T, n_embd] — latest merged h_final.
            visited_mask: [T] bool — positions already written by some interleaved pass (skip).
        """
        unrefreshed = ~visited_mask
        if not unrefreshed.any():
            return

        h_shifted = self._shift_right(h_final)  # [B, T, n_embd]
        T_full = h_shifted.size(1)
        unrefreshed_4d_f = unrefreshed.to(h_shifted.dtype).view(1, -1, 1, 1)
        unrefreshed_4d_b = unrefreshed.view(1, -1, 1, 1)  # bool, for torch.where

        # Rotary at positions 0..T_full-1. There's no KV cache prefix here —
        # the full pass runs at absolute positions 0..T-1.
        cos = self.cos[:, :T_full]
        sin = self.sin[:, :T_full]

        for i in range(self.config.n_layer):
            gate_fb = gate_buffer[i]
            if gate_fb is None:
                continue

            # ---- V refresh ----
            v_fb_new = self.interleaved_projections[str(i)](h_shifted)
            v_old = v_fb_old[i]
            v_diff = (v_fb_new - v_old) if isinstance(v_old, torch.Tensor) else v_fb_new
            v_delta = gate_fb.unsqueeze(-1) * v_diff * unrefreshed_4d_f

            kv_k, kv_v = kv_buffer[i]
            new_v = kv_v + v_delta

            # ---- K refresh ----
            if k_prerot_buffer[i] is not None:
                k_fb_new = self.interleaved_key_projections[str(i)](h_shifted)
                k_old = k_fb_old[i]
                k_diff = (k_fb_new - k_old) if isinstance(k_old, torch.Tensor) else k_fb_new

                # Add delta to pre-rotary K (full-T; mask applied via multiplication)
                k_prerot_delta = gate_fb.unsqueeze(-1) * k_diff * unrefreshed_4d_f
                new_k_prerot = k_prerot_buffer[i] + k_prerot_delta
                # Re-apply rotary + RMSNorm (same ops as forward)
                new_k_post = norm(apply_rotary_emb(new_k_prerot, cos, sin))
                # At visited positions keep the existing (interleaved-pass-written) K
                new_k = torch.where(unrefreshed_4d_b, new_k_post, kv_k)

                k_prerot_buffer[i] = new_k_prerot
                k_fb_old[i] = k_fb_new
                kv_buffer[i] = (new_k, new_v)
            else:
                kv_buffer[i] = (kv_k, new_v)

            v_fb_old[i] = v_fb_new

    def forward_interleaved(self, idx_interleaved, targets_interleaved, interleaved_positions, kv_buffer,
                       h_final_prev=None, loss_reduction='mean', pass_idx=0):
        """
        Interleaved forward pass: compute at interleaved positions only, using and updating buffered KV.
        Fresh K,V are written back to kv_buffer (mutated in-place at list level) so the next
        interleaved pass sees the latest KV. Gradients flow through the buffer (not detached).

        Args:
            idx_interleaved: [B, T_interleaved] token ids at interleaved positions
            targets_interleaved: [B, T_interleaved] targets at interleaved positions
            interleaved_positions: [T_interleaved] original position indices
            kv_buffer: list of (K, V) per layer, each [B, T_full, H_kv, D] — MUTATED
            h_final_prev: [B, T_interleaved, D] feedback hidden states at interleaved positions (already shifted)
            loss_reduction: 'mean' or 'none'
            pass_idx: current pass index (0-based; >= 1 for interleaved passes)

        Returns:
            (loss, h_final_interleaved)
        """
        B, T_interleaved = idx_interleaved.size()
        head_dim = self.config.n_embd // self.config.n_head

        # Rotary embeddings at original positions
        cos = self.cos[:, interleaved_positions]
        sin = self.sin[:, interleaved_positions]
        cos_sin = (cos, sin)

        # Embedding
        x = self.transformer.wte(idx_interleaved)
        x = norm(x)
        x0 = x if self.config.use_x0_residual else None

        for i, block in enumerate(self.transformer.h):
            # Residual lambdas, x0 residual, and hidden feedback injection
            if self.config.use_resid_lambdas:
                x = self.resid_lambdas[i] * x
            if self.config.use_x0_residual:
                x = x + self.x0_lambdas[i] * x0
            x = x + self.interleaved_lambdas[i] * h_final_prev

            layer_v_fb = self.interleaved_projections[str(i)](h_final_prev)
            layer_k_fb = self.interleaved_key_projections[str(i)](h_final_prev)
            layer_v_ve = self.interleaved_init_value_embeds[str(i)](idx_interleaved).view(B, T_interleaved, self.config.n_kv_head, head_dim)

            kv_k, kv_v = kv_buffer[i]
            x, kv_k_new, kv_v_new = block.forward_interleaved(
                x, None, cos_sin, self.window_sizes[i], kv_k, kv_v, interleaved_positions,
                v_fb=layer_v_fb, v_ve=layer_v_ve, k_fb=layer_k_fb, pass_idx=pass_idx)
            kv_buffer[i] = (kv_k_new, kv_v_new)  # update buffer for next interleaved pass

        h_final = norm(x)

        # LM head and loss
        softcap = 15
        logits = self.lm_head(h_final)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets_interleaved is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_interleaved.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss, h_final
        else:
            return logits

    def forward_interleaved_train(self, idx, targets, loss_reduction='mean', use_compiled_model=None):
        """
        Interleaved training step: 1 full forward pass (capturing every layer's K,V into a
        buffer) followed by config.interleaved_num_passes interleaved passes. Interleaved pass i
        recomputes the positions with pos % N == i against the shared KV buffer with
        fresh recurrent feedback. The loss is the mean over all passes.

        Args:
            idx: Input token ids [B, T]
            targets: Target token ids [B, T]
            loss_reduction: 'mean' or 'none'
            use_compiled_model: Optional compiled model (unused for the full pass here
                                since KV capture needs attribute access on the uncompiled
                                module; kept for interface compatibility)

        Returns:
            dict with:
              'loss': scalar training loss (mean over passes)
              'per_pass_losses': list of losses, [full, interleaved_1, ..., interleaved_N]
              'per_pass_masks': list aligned with per_pass_losses; None = loss covers all
                                positions, else a [T] bool mask of loss positions (used by eval)
              'h_final': final merged feedback hidden states (detached)
        """
        assert self.config.use_interleaved, "forward_interleaved_train requires use_interleaved=True"
        n_interleaved_passes = self.config.interleaved_num_passes
        assert n_interleaved_passes >= 1, f"interleaved_num_passes must be >= 1, got {n_interleaved_passes}"

        per_pass_losses = []
        per_pass_masks = []
        T_full = idx.size(1)
        pos_all = torch.arange(T_full, device=idx.device)
        kv_refresh = self.config.interleaved_kv_refresh

        # ---- Full pass (pass 0): capture the KV buffer ----
        # Feedback starts at zeros (the init value embeds ride the separate v_ve channel).
        # Passing an explicit zeros tensor (not None) keeps shapes static for torch.compile.
        h_recurrent = self._shift_right(self._make_zero_h_final(idx))
        if kv_refresh:
            fwd_result, kv_buffer, gate_buffer, k_prerot_buffer = self._forward_with_kv_capture(
                idx, targets, h_final_prev=h_recurrent,
                loss_reduction=loss_reduction, pass_idx=0,
                capture_gate_fb=True)
            # Full-pass v_fb / k_fb contributions are exactly zero (projection of zero
            # feedback); delta-track from the sentinel 0.
            v_fb_old = [0 for _ in range(self.config.n_layer)]
            k_fb_old = [0 for _ in range(self.config.n_layer)]
            visited_mask = torch.zeros(T_full, dtype=torch.bool, device=idx.device)
        else:
            fwd_result, kv_buffer = self._forward_with_kv_capture(
                idx, targets, h_final_prev=h_recurrent,
                loss_reduction=loss_reduction, pass_idx=0)
        loss, h_final = fwd_result
        per_pass_losses.append(loss)
        per_pass_masks.append(None)  # full pass covers all positions

        # Initial KV refresh: after the full pass, the captured K,V buffer has
        # gate_fb * {v,k}_fb (= 0, since full-pass feedback is zero) baked in everywhere.
        # The h_final from the full pass provides much richer feedback than zeros.
        # Refresh ALL positions before the first interleaved pass reads the buffer.
        if kv_refresh:
            self._refresh_kv_buffer(kv_buffer, gate_buffer, k_prerot_buffer,
                                    v_fb_old, k_fb_old, h_final, visited_mask)

        # ---- Interleaved passes: interleaved partial forwards against the KV buffer ----
        # Interleaved pass i covers positions with pos % n_interleaved_passes == i. Fresh K,V are
        # written back to kv_buffer by forward_interleaved; h_final is merged after each pass
        # so the next pass gets the latest feedback.
        for interleaved_idx in range(n_interleaved_passes):
            interleaved_mask = (pos_all % n_interleaved_passes) == interleaved_idx
            interleaved_positions = pos_all[interleaved_mask]
            idx_interleaved = idx[:, interleaved_mask]
            targets_interleaved = targets[:, interleaved_mask]

            # Compute feedback at interleaved positions from the LATEST h_final (kept fresh)
            h_recurrent_interleaved = self._shift_right(h_final)[:, interleaved_mask]

            # Use the compiled interleaved forward when installed (base_train --compile-interleaved);
            # interleaved positions are deterministic per pass -> static shapes.
            _fwd_interleaved = getattr(self, "_compiled_forward_interleaved", None) or self.forward_interleaved
            loss, h_final_interleaved = _fwd_interleaved(
                idx_interleaved, targets_interleaved, interleaved_positions, kv_buffer,
                h_final_prev=h_recurrent_interleaved, loss_reduction=loss_reduction,
                pass_idx=1 + interleaved_idx)

            # Merge interleaved h_final back into full h_final (keep feedback fresh for next interleaved pass)
            h_final = h_final.clone()
            h_final[:, interleaved_mask] = h_final_interleaved

            # Cheap V/K buffer refresh at never-visited positions (skipped after the final
            # interleaved pass, since nothing reads the buffer after that). This pass's written
            # positions are added to visited_mask BEFORE refresh so the refresh skips them
            # (forward_interleaved already wrote a complete fresh V there).
            if kv_refresh:
                visited_mask = visited_mask | interleaved_mask
                if interleaved_idx < n_interleaved_passes - 1:
                    self._refresh_kv_buffer(kv_buffer, gate_buffer, k_prerot_buffer,
                                            v_fb_old, k_fb_old, h_final, visited_mask)

            # For loss_reduction='none', scatter interleaved losses back to full-size tensor
            # so all per_pass_losses have the same shape [B*T] for aggregation
            if loss_reduction == 'none':
                B = idx.size(0)
                full_loss = torch.zeros(B * T_full, device=loss.device, dtype=loss.dtype)
                interleaved_flat = interleaved_mask.unsqueeze(0).expand(B, -1).reshape(-1)
                full_loss[interleaved_flat] = loss
                loss = full_loss

            per_pass_losses.append(loss)
            per_pass_masks.append(interleaved_mask)  # loss positions for this interleaved pass

        total_loss = sum(per_pass_losses) / len(per_pass_losses)

        return {
            'loss': total_loss,
            'per_pass_losses': per_pass_losses,
            'per_pass_masks': per_pass_masks,
            'h_final': h_final.detach() if h_final is not None else None,
        }

    @torch.no_grad()
    def print_interleaved_diagnostics(self, batches=None, n_steps=4):
        """Print per-layer interleaved diagnostics: λ values and average full-pass gate values.
        Call at validation time. batches/n_steps are used to compute gate averages."""
        if not self.config.use_interleaved:
            return

        # --- Print learnable λ values ---
        print0("[Interleaved Diagnostics] Shared lambdas:")
        for i in range(self.config.n_layer):
            parts = []
            if hasattr(self, 'resid_lambdas'):
                parts.append(f"α={self.resid_lambdas[i].item():.4f}")
            if hasattr(self, 'x0_lambdas'):
                parts.append(f"β={self.x0_lambdas[i].item():.4f}")
            parts.append(f"γ={self.interleaved_lambdas[i].item():.4f}")
            print0(f"  Layer {i:2d}: {', '.join(parts)}")

        # --- Compute average full-pass gate values over validation batches ---
        if batches is not None:
            # Enable gate collection on all attention modules
            for block in self.transformer.h:
                block.attn._collect_gate_stats = True
                block.attn._gate_sums = {}   # {pass_idx: sum}
                block.attn._gate_counts = {}  # {pass_idx: count}
                block.attn._gate_local_sums = {}  # {pass_idx: sum}

            batch_iter = iter(batches)
            for _ in range(n_steps):
                x, y = next(batch_iter)
                self.forward_interleaved_train(x, y, loss_reduction='mean')

            # Print full-pass gate stats and clean up. (Interleaved passes run through
            # forward_interleaved which doesn't collect gate stats; pass 0 is representative.)
            print0(f"[Interleaved Diagnostics] Average triple gate values — full pass (over {n_steps} batches):")
            for i, block in enumerate(self.transformer.h):
                attn = block.attn
                if 0 in attn._gate_sums and attn._gate_counts[0] > 0:
                    avg_gate_fb = attn._gate_sums[0] / attn._gate_counts[0]
                    avg_gate_local = attn._gate_local_sums[0] / attn._gate_counts[0]
                    avg_gate_ve = attn._gate_ve_sums[0] / attn._gate_counts[0]
                    print0(f"  Layer {i:2d}: gate_local={avg_gate_local:.4f}, gate_fb={avg_gate_fb:.4f}, gate_ve={avg_gate_ve:.4f}")
            # Clean up
            for block in self.transformer.h:
                block.attn._collect_gate_stats = False
                if hasattr(block.attn, '_gate_ve_sums'):
                    del block.attn._gate_ve_sums
                del block.attn._gate_sums, block.attn._gate_counts, block.attn._gate_local_sums

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
