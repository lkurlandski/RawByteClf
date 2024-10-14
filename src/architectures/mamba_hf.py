"""PyTorch MAMBA model."""

import math
from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple, Union, Literal

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, functional as F

from transformers.generation import GenerationMixin
from transformers.models.mamba.modeling_mamba import (
    MambaRMSNorm,
    MambaBlock,
    MambaPreTrainedModel,
    MambaCache,
    MambaOutput,
    MambaCausalLMOutput,
)
from transformers.configuration_utils import PretrainedConfig


# Overrridden to include the keys_to_ignore_at_inference attribute which is used
# to filter out the MambaCache object and prevent it from causing problems in accelerate.
class MambaConfig(PretrainedConfig):

    model_type = "mamba"
    keys_to_ignore_at_inference = ["cache_params"]

    def __init__(
        self,
        vocab_size: int = 50280,
        embedding_size: int = -1,
        hidden_size: int = 768,
        mlp_hidden_size: int = -1,
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
        mode: Literal["uni", "bi"] = "uni",
        tie_directions: bool = False,
        **kwargs,
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.embedding_size = embedding_size if embedding_size > 0 else hidden_size
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
        self.mode = mode
        self.tie_directions = tie_directions
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)


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

    def load_hook(self, state_dict, prefix, *args):
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
            hidden_states=all_hidden_states,
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

        if config.tie_directions:
            self.tie_forward_and_backward_weights()

        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def tie_forward_and_backward_weights(self):
        for i in range(self.config.num_hidden_layers):
            self._tie_or_clone_weights(self.layers_forw[i].mixer.in_proj, self.layers_back[i].mixer.in_proj)
            self._tie_or_clone_weights(self.layers_forw[i].mixer.out_proj, self.layers_back[i].mixer.out_proj)
            self._tie_or_clone_weights(self.layers_forw[i].mixer.x_proj, self.layers_back[i].mixer.x_proj)

    def check_shared_weights(self):
        if not self.config.tie_directions:
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
        all_hidden_states = () if output_hidden_states else None
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
            hidden_states = hidden_states_forw + hidden_states_back.flip(1)
            hidden_states_forw = hidden_states
            hidden_states_back = hidden_states

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
            hidden_states=all_hidden_states,
        )


class MambaForCausalLM(MambaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MambaConfig):
        if config.mode == "bi":
            raise ValueError("MambaForCausalLM does not support bidirectional models.")

        super().__init__(config)
        self.config: MambaConfig

        self.backbone = MambaModel(config)
        self.embedding_out_projection = nn.Linear(config.hidden_size, config.embedding_size) if self.config.hidden_size != self.config.embedding_size else nn.Identity()
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: MambaOutput, model_kwargs: Dict[str, Any], num_new_tokens: int = 1, **kwargs
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
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

        hidden_states = mamba_outputs[0]
        hidden_states = self.embedding_out_projection(hidden_states)
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )


class MambaForMaskedLM(MambaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MambaConfig):
        if config.mode == "uni":
            raise ValueError("MambaForCausalLM does not support unidirectional models.")

        super().__init__(config)
        self.config: MambaConfig

        self.backbone = BiMambaModel(config)
        self.embedding_out_projection = nn.Linear(config.hidden_size, config.embedding_size) if self.config.hidden_size != self.config.embedding_size else nn.Identity()
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

        hidden_states = mamba_outputs[0]
        hidden_states = self.embedding_out_projection(hidden_states)
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )


class MambaForSequenceClassification(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.config: MambaConfig

        if self.config.mode == "uni":
            self.backbone = MambaModel(config)
        elif self.config.mode == "bi":
            self.backbone = BiMambaModel(config)
        else:
            raise ValueError(f"Invalid mode: {self.config.mode}")

        if self.config.mlp_hidden_size > 0:
            self.clf_neck = nn.Linear(config.hidden_size, config.mlp_hidden_size)
            self.clf_actv = F.leaky_relu
            self.clf_head = nn.Linear(config.mlp_hidden_size, config.num_labels)
        else:
            self.clf_neck = nn.Identity()
            self.clf_actv = nn.Identity()
            self.clf_head = nn.Linear(config.hidden_size, config.num_labels)

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
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

        hidden_states = mamba_outputs[0]

        logits = self.clf_neck(hidden_states)
        logits = self.clf_actv(logits)
        logits = self.clf_head(logits)

        batch_size = input_ids.shape[0]
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)
        clf_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(clf_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(clf_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(clf_logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(clf_logits, labels)

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=clf_logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )
