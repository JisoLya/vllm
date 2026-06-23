# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Domino speculative decoding model, built on top of DFlash.

Domino extends DFlash's pure parallel drafting with a lightweight serial
refinement step inside each drafted block: a single-layer GRU encodes the
already-realized prefix (bonus token + previously accepted draft tokens
within the block) and the resulting state is fused with the parallel
attention's per-position hidden state to produce a *bias* that is added to
the block's base logits before sampling. This recovers some of the
intra-block dependency information that pure parallel drafting discards,
at the cost of a short serial loop over the block.

Architecturally this file mirrors `qwen3_dflash.py`:
- Context K/V (target model hidden states) are pre-projected and written
  directly into the draft model's KV cache via `precompute_and_store_context_kv`,
  exactly as in DFlash.
- The decoder stack (`Qwen3DominoAttention`/`Qwen3DominoDecoderLayer`/
  `Qwen3DominoModel`) is identical to DFlash's: one non-causal forward pass
  produces `parallel_hiddens` for every query position in the block.
- What's added on top is the GRU refinement path
  (`Qwen3DominoRefineStep`, `Qwen3DominoModel.refine_step_forward`) that a
  proposer (e.g. `DominoProposer`) calls once per block position, in a
  loop dispatched through vLLM's standard `CudagraphDispatcher` /
  `set_forward_context` machinery — the same mechanism `DFlashProposer`'s
  main forward and EAGLE's multi-step loop already use — since this
  step is inherently serial and cannot be parallelized like the main
  attention forward.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


def is_domino_projector(projector_type: str | None) -> bool:
    return projector_type == "domino"


class Qwen3DominoAttention(nn.Module):
    """Attention for Domino speculative decoding.

    Identical in spirit to DFlashQwen3Attention: context K/V (from the
    target model's hidden states) are pre-inserted into the KV cache before
    the forward pass; this layer only computes attention for the query
    (block) tokens.
    Adapted from Qwen3Attention / DFlashQwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,  # Domino has o_proj bias when using attention bias
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Domino attention assumes that the KV cache is already populated
        with the context K/V from the target model's hidden states. This
        forward op computes attention for the query (block) tokens only.
        See also: precompute_and_store_context_kv"""
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DominoDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER

        self.self_attn = Qwen3DominoAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DominoPrefixGRUCell(nn.Module):
    """Single-step GRU cell used to encode the realized block prefix.

    Mirrors `nn.GRU(num_layers=1, bias=False)` from the reference
    implementation, but exposes an explicit single-step `forward` so the
    proposer can call it once per realized token inside the block loop
    without going through a variable-length `nn.GRU` sequence call (which
    is awkward to make CUDA-graph-friendly). Mathematically equivalent to
    `nn.GRU` with `bias=False`, and can load the same checkpoint weights
    (`weight_ih_l0`, `weight_hh_l0`).
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Stacked [reset, update, new] gates, matching nn.GRU's layout.
        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))

    def init_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(
        self, input_embed: torch.Tensor, prev_hidden: torch.Tensor
    ) -> torch.Tensor:
        """One GRU step.

        Args:
            input_embed: [batch, input_size]
            prev_hidden: [batch, hidden_size]
        Returns:
            new_hidden: [batch, hidden_size]
        """
        gi = F.linear(input_embed, self.weight_ih_l0)
        gh = F.linear(prev_hidden, self.weight_hh_l0)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)

        new_hidden = (1.0 - update_gate) * new_gate + update_gate * prev_hidden
        return new_hidden


@support_torch_compile
class Qwen3DominoRefineStep(nn.Module):
    """One step of Domino's block-internal serial GRU refinement.

    This is split out from `Qwen3DominoModel` into its own
    `@support_torch_compile`-decorated module specifically so it can be
    dispatched through vLLM's standard `CudagraphDispatcher` / PIECEWISE
    CUDA graph machinery — the same mechanism `DFlashProposer` (and EAGLE's
    multi-step loop in `SpecDecodeBaseProposer.propose`) already uses for
    the main per-step forward pass. Capturing a fixed-size, fixed-shape
    `forward(token_ids, parallel_hidden, base_logits, gru_hidden)` call
    here means the proposer's refinement loop is just an ordinary
    `set_forward_context(...): self.refine_step(**kwargs)` call per
    iteration, exactly like every other multi-step drafter in vLLM,
    instead of a hand-rolled, separately-managed `torch.cuda.CUDAGraph`.

    One call here corresponds to one iteration of the reference
    implementation's loop body:

        z_i = parallel_hiddens[:, i, :]
        s_i = gru_hidden
        bias = self.embed_proj(torch.cat([z_i, s_i], dim=-1))
        current_token_id = sample(base_logits[:, i, :] + bias, temperature)
        new_embed = target.model.embed_tokens(current_token_id)
        _, gru_hidden = self.prefix_gru(new_embed, gru_hidden)

    except greedy-only (argmax) sampling, and with `token_ids` as an input
    (the token whose embedding should advance the GRU state — i.e. the
    *previous* step's sampled token, or the bonus token for step 0) rather
    than computed from a token sampled inside this same call. This input/
    output split keeps the module a pure function of fixed-shape tensors,
    which is what graph capture requires.

    Implementation note: this module does *not* hold its weight-bearing
    submodules (the GRU, the projector linears, the embedding table) as
    ordinary `nn.Module` attributes. Doing so would register them a
    second time under `self.refine_step.*` in `named_parameters()` — since
    they're the *same* `nn.Parameter` objects already registered under
    `Qwen3DominoModel`'s own attributes — which would make `load_weights`
    (and any sanity check that expects each on-disk weight to map to
    exactly one parameter name) see an unloaded duplicate. Instead, the
    underlying weight tensors are kept as plain (non-`nn.Module`) Python
    attributes set via `object.__setattr__`, and this module computes with
    them directly via functional ops (`F.linear`, `F.embedding`, the GRU
    math). The actual `nn.Parameter`s are owned solely by
    `Qwen3DominoModel`.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        gru: DominoPrefixGRUCell,
        embed_tokens: VocabParallelEmbedding,
        embed_proj_in: ReplicatedLinear,
        embed_proj_act: nn.Module,
        embed_proj_out: ReplicatedLinear,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Hold references via object.__setattr__ so nn.Module.__setattr__
        # does not register these as submodules (see class docstring).
        # We still keep `embed_proj_act` (stateless, e.g. nn.SiLU) as a
        # normal attribute since it owns no parameters and registering it
        # twice is harmless.
        object.__setattr__(self, "_gru", gru)
        object.__setattr__(self, "_embed_tokens", embed_tokens)
        object.__setattr__(self, "_embed_proj_in", embed_proj_in)
        object.__setattr__(self, "_embed_proj_out", embed_proj_out)
        self.embed_proj_act = embed_proj_act

    def forward(
        self,
        token_ids: torch.Tensor,
        parallel_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        gru_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the GRU by `token_ids` and sample the next position.

        Args:
            token_ids: [batch] int ids of the token to feed into the GRU
                this step (the previous position's realized token, or the
                block's bonus token on the first call).
            parallel_hidden: [batch, hidden_size] this position's hidden
                state from the parallel (non-causal) attention forward.
            base_logits: [batch, vocab_size] target lm_head logits applied
                to `parallel_hidden`.
            gru_hidden: [batch, gru_hidden_dim] GRU state *before* this
                step's token is folded in.

        Returns:
            (next_token_id, new_gru_hidden):
                next_token_id: [batch] greedily sampled token id for this
                    position (base_logits + GRU-derived bias).
                new_gru_hidden: [batch, gru_hidden_dim] GRU state after
                    folding in `token_ids`, ready for the next call.
        """
        token_embeds = self._embed_tokens(token_ids)
        new_gru_hidden = self._gru(token_embeds, gru_hidden)

        fused = torch.cat([parallel_hidden, new_gru_hidden], dim=-1)
        x = self._embed_proj_in(fused)
        x = self.embed_proj_act(x)
        bias = self._embed_proj_out(x)

        logits = base_logits + bias
        next_token_id = logits.argmax(dim=-1)
        return next_token_id, new_gru_hidden


@support_torch_compile
class Qwen3DominoModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))
        self.drafter_config = drafter_config

        projector_type = drafter_config.get("projector_type", None)
        if not is_domino_projector(projector_type):
            raise ValueError(
                "Qwen3DominoModel only supports projector_type='domino'; "
                f"got projector_type={projector_type!r}."
            )

        if "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                Qwen3DominoDecoderLayer(
                    current_vllm_config,
                    config=self.config,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        # --- Domino-specific refinement modules ---
        self.gru_hidden_dim = drafter_config["gru_hidden_dim"]
        self.emb_dim = drafter_config["emb_dim"]

        self.prefix_gru = DominoPrefixGRUCell(
            input_size=self.config.hidden_size,
            hidden_size=self.gru_hidden_dim,
        )

        embed_proj_in_dim = self.config.hidden_size + self.gru_hidden_dim
        self.embed_proj_in = ReplicatedLinear(
            input_size=embed_proj_in_dim,
            output_size=self.emb_dim,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_proj.0"),
            return_bias=False,
        )
        self.embed_proj_act = nn.SiLU()
        self.embed_proj_out = ReplicatedLinear(
            input_size=self.emb_dim,
            output_size=self.config.vocab_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_proj.2"),
            return_bias=False,
        )

        # Independently compiled/graphed entry point for one block-internal
        # refinement step. Shares weights with the modules above; see
        # Qwen3DominoRefineStep's docstring for why this needs to be a
        # separate @support_torch_compile module rather than a plain method.
        self.refine_step = Qwen3DominoRefineStep(
            vllm_config=vllm_config,
            gru=self.prefix_gru,
            embed_tokens=self.embed_tokens,
            embed_proj_in=self.embed_proj_in,
            embed_proj_act=self.embed_proj_act,
            embed_proj_out=self.embed_proj_out,
            prefix=maybe_prefix(prefix, "refine_step"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def init_gru_state(self, batch_size: int) -> torch.Tensor:
        """Returns a zero-initialized GRU hidden state for a new block."""
        param = self.prefix_gru.weight_hh_l0
        return self.prefix_gru.init_state(
            batch_size, device=param.device, dtype=param.dtype
        )

    def refine_step_forward(
        self,
        token_ids: torch.Tensor,
        parallel_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        gru_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one block-internal refinement step.

        Thin pass-through to `self.refine_step` (a separate
        `@support_torch_compile` module — see its docstring). Kept as a
        method here so the proposer's call sites read naturally as
        "the model does one refinement step", without needing to know
        that the computation actually lives in a sibling submodule for
        CUDA-graph-dispatch reasons.

        Args:
            token_ids: [batch] token id to feed into the GRU this step
                (previous position's realized token, or the block's bonus
                token on the first call).
            parallel_hidden: [batch, hidden_size] this position's hidden
                state from the parallel (non-causal) forward pass.
            base_logits: [batch, vocab_size] target lm_head logits applied
                to `parallel_hidden`.
            gru_hidden: [batch, gru_hidden_dim] GRU state before this
                step's token is folded in.
        Returns:
            (next_token_id, new_gru_hidden)
        """
        return self.refine_step(token_ids, parallel_hidden, base_logits, gru_hidden)

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once. Also aliases the weight of the hidden_norm.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights: list of [head_dim] tensors, one per layer.
        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        # Validation that RoPE params are the same across all layers
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for Domino precomputation"

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        # Validation that all layers have the same attention config
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for Domino precomputation"

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Precompute K/V for context states write them into each layer's KV cache.

        Input context states are projected to K/V, normed, and have RoPE applied.
        Since the context shape is different than the query shape, we can't rely on the
        regular forward pass to apply torch.compile and CUDA graphs to this section.
        As such, this function is optimized to minimize the number of torch ops present:
        we use fused vLLM kernels for RMSNorm and RoPE, fuse the GEMM into one
        large projection, and avoid cloning buffers (with .contiguous()) where possible.

        When context_slot_mapping is None (e.g. during dummy_run) only
        the computation runs, and no K/V is written to cache.
        """
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "Domino buffer initialization was skipped. If dummy weights are not "
                "in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

        # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            ops.rms_norm(
                all_k_normed[i],
                all_k[i],
                self._k_norm_weights[i],
                self._rms_norm_eps,
            )

        # --- Fused RoPE across all layers ---
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        # In-place RoPE: pass K as the "query" arg with key=None.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        for i in range(L):
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                context_slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One parallel (non-causal) forward over the block's query tokens.

        Produces `parallel_hiddens` for every query position, analogous to
        DFlash's forward. The block-internal GRU refinement (Domino's
        contribution on top of plain DFlash) is *not* performed here — it
        runs as a separate per-position loop via `refine_step_forward`
        (each call dispatched through the standard CUDA graph machinery,
        same as this forward), since the refinement is inherently serial
        across block positions.
        """
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        # Reference HF checkpoint stores the GRU and projector under
        # `prefix_gru.*` / `embed_proj.{0,2}.*`; remap to this module's names.
        renames = [
            ("prefix_gru.weight_ih_l0", "prefix_gru.weight_ih_l0"),
            ("prefix_gru.weight_hh_l0", "prefix_gru.weight_hh_l0"),
            ("embed_proj.0.weight", "embed_proj_in.weight"),
            ("embed_proj.2.weight", "embed_proj_out.weight"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for old_suffix, new_suffix in renames:
                if name.endswith(old_suffix):
                    name = name[: -len(old_suffix)] + new_suffix
                    break
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3DominoForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3DominoModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    # --- Domino refinement entry point, delegated to the inner model. ---
    # A DominoProposer is expected to call this once per block position,
    # in a loop dispatched through the same CudagraphDispatcher /
    # set_forward_context machinery used for the main forward pass —
    # see Qwen3DominoRefineStep's docstring for why this is its own
    # @support_torch_compile module rather than a plain Python method.

    def init_gru_state(self, batch_size: int) -> torch.Tensor:
        return self.model.init_gru_state(batch_size)

    def refine_step_forward(
        self,
        token_ids: torch.Tensor,
        parallel_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        gru_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.refine_step_forward(
            token_ids, parallel_hidden, base_logits, gru_hidden
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "Domino should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()