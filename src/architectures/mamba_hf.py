"""PyTorch MAMBA model."""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn


import transformers

# pylint: disable=wrong-import-position
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except (ImportError, ModuleNotFoundError) as _err:
    transformers.utils.import_utils.is_causal_conv1d_available = lambda: False
    print(f"{_err.__class__.__name__}: causal_conv1d")

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except (ImportError, ModuleNotFoundError) as _err:
    transformers.utils.import_utils.is_mamba_ssm_available = lambda: False
    print(f"{_err.__class__.__name__}: mamba_ssm")
# pylint: enable=wrong-import-position

from transformers.generation import GenerationMixin
from transformers.models.mamba.modeling_mamba import (  # pylint: disable=no-name-in-module
    MambaRMSNorm,
    MambaBlock,
    MambaPreTrainedModel as MambaPreTrainedModelHF,
    MambaCache as MambaCacheHF,
    MambaOutput,
    MambaCausalLMOutput,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import ModelOutput

from src.architectures.head_utils import Head, pool_logits, get_clf_loss, get_clm_loss, get_mlm_loss, check_tie_embeddings_will_work


ARG_REQUIRED = -1
ARG_INFERRED = -1


class MambaConfig(PretrainedConfig):

    model_type = "mamba"
    keys_to_ignore_at_inference = ["cache_params"]

    def __init__(
        self,
        vocab_size: int = ARG_REQUIRED,
        hidden_size: int = ARG_REQUIRED,
        embedding_size: int = ARG_INFERRED,
        head_hidden_size: int = 0,
        head_num_hidden_layers: int = 0,
        head_dropout: float = 0.1,
        state_size: int = 16,
        num_hidden_layers: int = 32,
        layer_norm_epsilon: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        expand: int = 2,
        conv_kernel: int = 4,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        initializer_range: float = 0.1,
        residual_in_fp32: bool = True,
        time_step_rank: str = "auto",
        time_step_scale: float = 1.0,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_init_scheme: str = "random",
        time_step_floor: float = 1e-4,
        rescale_prenorm_residual: bool = False,
        use_cache: bool = True,
        bi_tie_directions: bool = False,
        bi_mix_directions: bool = False,
        bi_add_directions: bool = False,
        use_mambapy: bool = False,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = hidden_size if embedding_size == ARG_INFERRED else embedding_size
        self.head_hidden_size = head_hidden_size
        self.head_num_hidden_layers = head_num_hidden_layers
        self.head_dropout = head_dropout
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.bi_tie_directions = bi_tie_directions
        self.bi_mix_directions = bi_mix_directions
        self.bi_add_directions = bi_add_directions
        self.use_mambapy = use_mambapy

        check_tie_embeddings_will_work(
            self.tie_word_embeddings,
            kwargs.get("num_labels"),
            self.hidden_size,
            self.embedding_size,
            self.head_hidden_size,
            self.head_num_hidden_layers,
        )


class MambaCache(MambaCacheHF):

    """Overridden Mamba cache fit for monkey patch.

      - The original implementation will raise an error when training in mixed precision
        because new_conv_state will always be float32. This is patched to convert the
        dtype of new_conv_state to whatever dtype is currently in use.
      - The original implementation will also raise errors when training with accelerate
        because of it tries to cast the object as a float and change its device to the cpu.
        This is patched to provide these methods and appear to manipulate the object, but
        actually does nothing.
    """

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        # The HF implementation does not work if new_conv_state is a different dtype
        new_conv_state = new_conv_state.to(self.conv_states[layer_idx].dtype)
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)
        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def float(self):
        self.dtype = torch.float32  # pylint: disable=attribute-defined-outside-init
        # self.conv_states = {k: v.float() for k, v in self.conv_states.items()}
        # self.ssm_states = {k: v.float() for k, v in self.ssm_states.items()}
        return self

    def detach(self):
        self.device = "cpu"  # pylint: disable=attribute-defined-outside-init
        # self.conv_states = {k: v.detach() for k, v in self.conv_states.items()}
        # self.ssm_states = {k: v.detach() for k, v in self.ssm_states.items()}
        return self


transformers.cache_utils.MambaCache = MambaCache  # pylint: disable=no-member


@dataclass
class MambaMaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MambaSequenceClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MambaPreTrainedModel(MambaPreTrainedModelHF):

    def save_pretrained(self, *args, **kwargs) -> None:
        if self.config.is_decoder or not self.config.bi_tie_directions:
            super().save_pretrained(*args, **kwargs)
            return

        if not hasattr(self, self.base_model_prefix):
            super().save_pretrained(*args, **kwargs)
            return

        print("Temporarily removing weight-tying to save model.")
        self.backbone.tie_forward_and_backward_weights(tie=False, clone=True)
        super().save_pretrained(*args, **kwargs)
        self.backbone.tie_forward_and_backward_weights(tie=True, clone=False)
        self.backbone.check_shared_weights()


transformers.models.mamba.modeling_mamba.MambaPreTrainedModel = MambaPreTrainedModel  # pylint: disable=no-member


class MambaModel(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.config: MambaConfig

        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_size, self.config.pad_token_id)
        self.embedding_projection = nn.Linear(config.embedding_size, config.hidden_size) if config.hidden_size != config.embedding_size else nn.Identity()
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):  # pylint: disable=unused-argument
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = MambaCache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = self.embedding_projection(inputs_embeds)
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
        )


class BiMambaModel(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.config: MambaConfig

        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_size, self.config.pad_token_id)
        self.embedding_projection = nn.Linear(config.embedding_size, config.hidden_size) if config.hidden_size != config.embedding_size else nn.Identity()
        self.layers_forw = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])
        self.layers_back = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        if config.bi_tie_directions:
            self.tie_forward_and_backward_weights(tie=True, clone=False)

        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()
        self.check_shared_weights()
        self._weights_checked = False

    def load_hook(self, state_dict, prefix, *args):  # pylint: disable=unused-argument
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    @staticmethod
    def tie_or_clone_projections(src: nn.Linear, dst: nn.Linear, *, tie: bool = None, clone: bool = None) -> None:
        if bool(tie) == bool(clone):
            raise ValueError("Exactly one of `tie` or `clone` must be True.")
        if tie:
            dst.weight = src.weight
            dst.bias   = src.bias
        elif clone:
            dst.weight = nn.Parameter(src.weight.clone())
            dst.bias   = nn.Parameter(src.bias.clone()) if src.bias is not None else None

    def tie_forward_and_backward_weights(self, tie: bool = None, clone: bool = None) -> None:
        for i in range(self.config.num_hidden_layers):
            BiMambaModel.tie_or_clone_projections(self.layers_forw[i].mixer.in_proj, self.layers_back[i].mixer.in_proj, tie=tie, clone=clone)
            BiMambaModel.tie_or_clone_projections(self.layers_forw[i].mixer.out_proj, self.layers_back[i].mixer.out_proj, tie=tie, clone=clone)
            BiMambaModel.tie_or_clone_projections(self.layers_forw[i].mixer.x_proj, self.layers_back[i].mixer.x_proj, tie=tie, clone=clone)

    def check_shared_weights(self):
        if not self.config.bi_tie_directions:
            return

        for i in range(self.config.num_hidden_layers):
            if not self.layers_forw[i].mixer.in_proj.weight.data_ptr() == self.layers_back[i].mixer.in_proj.weight.data_ptr():
                raise ValueError(f"Layer {i} in_proj weights are not shared")
            if not self.layers_forw[i].mixer.out_proj.weight.data_ptr() == self.layers_back[i].mixer.out_proj.weight.data_ptr():
                raise ValueError(f"Layer {i} out_proj weights are not shared")
            if not self.layers_forw[i].mixer.x_proj.weight.data_ptr() == self.layers_back[i].mixer.x_proj.weight.data_ptr():
                raise ValueError(f"Layer {i} x_proj weights are not shared")

    def prepare_input_for_backward_model(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Assumes the input is structured as follows:

        <bos> <token1> <token2> ... <eos> <pad> <pad>

        Using a batch size of 32, and a sequence length of 16384, this seemed to take about 0.0075s/call.
        this would result in about a 4 minute slow down to process 1M samples in batches of 32.
        """
        # t_0 = time.time()

        PAD_TOKEN_ID = self.config.pad_token_id
        BOS_TOKEN_ID = self.config.bos_token_id
        EOS_TOKEN_ID = self.config.eos_token_id

        assert len(set([PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID])) == 3

        DEVICE = input_ids.device
        DTYPE = input_ids.dtype
        B = input_ids.shape[0]
        L = input_ids.shape[1]

        reversed_input_ids = torch.zeros_like(input_ids)

        for i in range(B):
            pad_idx = torch.nonzero(torch.eq(input_ids[i], PAD_TOKEN_ID), as_tuple=False)
            pad_idx = L if len(pad_idx) == 0 else pad_idx[0].to("cpu").item()  # index of first pad token

            start = 1
            end = pad_idx - 1

            tensors = []
            tensors.append(torch.tensor([BOS_TOKEN_ID], device=DEVICE, dtype=DTYPE))
            tensors.append(input_ids[i][start:end].flip(0))
            tensors.append(torch.tensor([EOS_TOKEN_ID], device=DEVICE, dtype=DTYPE))
            if (length := sum(x.shape[0] for x in tensors)) < L:
                tensors.append(torch.full((L - length,), PAD_TOKEN_ID, device=DEVICE, dtype=DTYPE))

            reversed_input_ids[i] = torch.cat(tensors)

        # print(f"Reversed {B} inputs in: {time.time() - t_0:.4f}s")

        return reversed_input_ids

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[tuple[MambaCache, MambaCache]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[tuple[torch.LongTensor, torch.LongTensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MambaOutput]:
        # Check the shared weights.
        if not self._weights_checked:
            self.check_shared_weights()
            self._weights_checked = True

        if attention_mask is not None:
            raise NotImplementedError("Attention mask is not implemented for the backward models.")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None or inputs_embeds is not None:
            raise ValueError("Bidirectional Mamba needs input_ids to be passed.")

        input_ids_forw = input_ids
        input_ids_back = self.prepare_input_for_backward_model(input_ids)

        inputs_embeds_forw: torch.Tensor = self.embeddings(input_ids_forw)
        inputs_embeds_back: torch.Tensor = self.embeddings(input_ids_back)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = (
                    MambaCache(
                        self.config,
                        inputs_embeds_forw.size(0),
                        device=inputs_embeds_forw.device,
                        dtype=inputs_embeds_forw.dtype,
                    ),
                    MambaCache(
                        self.config,
                        inputs_embeds_back.size(0),
                        device=inputs_embeds_back.device,
                        dtype=inputs_embeds_back.dtype,
                    )
                )
                cache_position = (
                    torch.arange(0, self.config.conv_kernel, device=inputs_embeds_forw.device),
                    torch.arange(0, self.config.conv_kernel, device=inputs_embeds_back.device),
                )
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states_forw: torch.Tensor = self.embedding_projection(inputs_embeds_forw)
        hidden_states_back: torch.Tensor = self.embedding_projection(inputs_embeds_back)
        all_hidden_states_forw = () if output_hidden_states else None
        all_hidden_states_back = () if output_hidden_states else None
        for mixer_block_forw, mixer_block_back in zip(self.layers_forw, self.layers_back):
            cache_params_forw = cache_params[0] if cache_params is not None else None
            cache_params_back = cache_params[1] if cache_params is not None else None
            cache_position_forw = cache_position[0] if cache_position is not None else None
            cache_position_back = cache_position[1] if cache_position is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states_forw = self._gradient_checkpointing_func(
                    mixer_block_forw.__call__, hidden_states_forw, cache_params_forw, cache_position_forw, attention_mask
                )
            else:
                hidden_states_forw = mixer_block_forw(
                    hidden_states_forw,
                    cache_params=cache_params_forw,
                    cache_position=cache_position_forw,
                    attention_mask=attention_mask,
                )

            if self.gradient_checkpointing and self.training:
                hidden_states_back = self._gradient_checkpointing_func(
                    mixer_block_back.__call__, hidden_states_back, cache_params_back, cache_position_back, attention_mask
                )
            else:
                hidden_states_back = mixer_block_back(
                    hidden_states_back,
                    cache_params=cache_params_back,
                    cache_position=cache_position_back,
                    attention_mask=attention_mask,
                )

            # Flipping the backward hidden states aligns them with the forward ones.
            if self.config.bi_mix_directions:
                hidden_states = hidden_states_forw + hidden_states_back.flip(1)
                hidden_states_forw = hidden_states
                hidden_states_back = hidden_states

            if output_hidden_states:
                all_hidden_states_forw = all_hidden_states_forw + (hidden_states_forw,)
                all_hidden_states_back = all_hidden_states_back + (hidden_states_back,)

        hidden_states_forw = self.norm_f(hidden_states_forw)  # (B, T, H)
        hidden_states_back = self.norm_f(hidden_states_back)  # (B, T, H)

        if output_hidden_states:
            all_hidden_states_forw = all_hidden_states_forw + (hidden_states_forw,)
            all_hidden_states_back = all_hidden_states_back + (hidden_states_back,)

        if not return_dict:
            return (
                (hidden_states_forw, hidden_states_back),
                (cache_params_forw, cache_params_back) if use_cache else None,
                (all_hidden_states_forw, all_hidden_states_back) if output_hidden_states else None,
            )

        return MambaOutput(
            last_hidden_state=(hidden_states_forw, hidden_states_back),
            cache_params=(cache_params_forw, cache_params_back) if use_cache else None,
        )


class MambaForCausalLM(MambaPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["head_clm.final_layer.weight", "head_clm.final_layer.bias"]

    def __init__(self, config: MambaConfig):
        if not config.is_decoder:
            raise ValueError("MambaForCausalLM does not support bidirectional models.")

        super().__init__(config)
        self.config: MambaConfig

        self.backbone = MambaModel(config)
        self.head_clm = Head(
            config.hidden_size,
            config.vocab_size,
            config.head_hidden_size,
            config.head_num_hidden_layers,
            config.head_dropout,
        )

        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.head_clm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.head_clm.set_output_embeddings(new_embeddings)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: MambaOutput, model_kwargs: Dict[str, Any], num_new_tokens: int = 1, **kwargs  # pylint: disable=unused-argument
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Overwitten -- uses `cache_params` as opposed to `past_key_values`

        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1].unsqueeze(-1)

                if attention_mask is not None:
                    attention_mask = None

            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: MambaOutput = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        logits = self.head_clm(outputs.last_hidden_state)
        logits = pool_logits("none", logits, input_ids, self.config.pad_token_id)
        loss = get_clm_loss(logits, labels, self.config.vocab_size) if labels is not None else None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(loss=loss, logits=logits)


class MambaForMaskedLM(MambaPreTrainedModel):

    _tied_weights_keys = ["head_mlm.final_layer.weight", "head_mlm.final_layer.bias"]

    def __init__(self, config: MambaConfig):
        if config.is_decoder:
            raise ValueError("MambaForCausalLM does not support unidirectional models.")

        super().__init__(config)
        self.config: MambaConfig

        self.backbone = BiMambaModel(config)
        self.head_mlm = Head(
            config.hidden_size if (config.is_decoder or config.bi_add_directions) else config.hidden_size * 2,
            config.vocab_size,
            config.head_hidden_size,
            config.head_num_hidden_layers,
            config.head_dropout,
        )

        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.head_mlm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.head_mlm.set_output_embeddings(new_embeddings)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Union[Tuple, MambaMaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: MambaOutput = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

        # For classification, we want the hidden states corresponding to each token,
        # which is why this code DOES flip the backward hidden states along the sequence axis.
        if isinstance(self.backbone, BiMambaModel):
            hidden_states_forw, hidden_states_back = outputs.last_hidden_state
            if self.config.bi_add_directions:
                hidden_states = hidden_states_forw + hidden_states_back.flip(1)
            else:
                hidden_states = torch.cat([hidden_states_forw, hidden_states_back.flip(1)], dim=2)
        else:
            hidden_states = outputs.last_hidden_state

        logits = self.head_mlm(hidden_states)
        logits = pool_logits("none", logits, input_ids, self.config.pad_token_id)
        loss = get_mlm_loss(logits, labels, self.config.vocab_size) if labels is not None else None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaMaskedLMOutput(loss=loss, logits=logits)


class MambaForSequenceClassification(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.config: MambaConfig

        self.backbone = MambaModel(config) if config.is_decoder else BiMambaModel(config)
        self.head_clf = Head(
            config.hidden_size if (config.is_decoder or config.bi_add_directions) else config.hidden_size * 2,
            config.num_labels,
            config.head_hidden_size,
            config.head_num_hidden_layers,
            config.head_dropout,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Union[Tuple, MambaSequenceClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: MambaOutput = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

        # For classification, we want the final hidden states for the forward and backward models,
        # which is why this code DOES NOT flip the backward hidden states along the sequence axis.
        if isinstance(self.backbone, BiMambaModel):
            hidden_states_forw, hidden_states_back = outputs.last_hidden_state
            if self.config.bi_add_directions:
                hidden_states = hidden_states_forw + hidden_states_back
            else:
                hidden_states = torch.cat([hidden_states_forw, hidden_states_back], dim=2)
        else:
            hidden_states = outputs.last_hidden_state

        logits = self.head_clf.forward(hidden_states)
        logits = pool_logits("last", logits, input_ids, self.config.pad_token_id)
        loss = get_clf_loss(logits, labels, self.config.num_labels, self.config.problem_type) if labels is not None else None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaSequenceClassificationOutput(loss=loss, logits=logits)
