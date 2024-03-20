"""
Huggingface compatible implementation of Mamba.
"""

import math
from typing import Optional, Literal

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, MaskedLMOutput, SequenceClassifierOutput
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch import LongTensor, Tensor

from mamba_ssm.modules.mamba_simple import Block, Mamba
from mamba_ssm.models.mixer_seq_simple import create_block
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


def prepare_input_for_backward_model(
    input_ids: LongTensor,
    pad_token_id: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    padding_side: Optional[Literal["right", "left"]] = "right",
) -> LongTensor:
    """Prepares a batch of inputs for the 'backwards' model in bidirectional recurrent architecture.

    Args:
        input_ids (LongTensor): Batch of inputs for forward model.
        pad_token_id (int): Pad token.
        bos_token_id (Optional[int], optional): If provided, assumes sequences are preceeded by this
            token. Defaults to None.
        eos_token_id (Optional[int], optional): If provided, assumes sequences are succeeded by this
            token. Defaults to None.

    Returns:
        LongTensor: Batch of inputs for backward model.
    """
    if padding_side is not None and padding_side != "right":
        raise NotImplementedError(f"{padding_side=}")

    DEVICE = input_ids.device
    DTYPE = input_ids.dtype
    B = input_ids.shape[0]
    L = input_ids.shape[1]

    reversed_input_ids = torch.zeros_like(input_ids)

    for i in range(B):
        pad_idx = torch.nonzero(torch.eq(input_ids[i], pad_token_id), as_tuple=False)
        pad_idx = L if len(pad_idx) == 0 else pad_idx[0].to("cpu").item()  # index of first pad token

        start = 0 if bos_token_id is None else 1
        end = pad_idx if eos_token_id is None else pad_idx - 1

        tensors = []
        if bos_token_id is not None:
            tensors.append(torch.tensor([bos_token_id], device=DEVICE, dtype=DTYPE))
        tensors.append(input_ids[i][start:end].flip(0))
        if eos_token_id is not None:
            tensors.append(torch.tensor([eos_token_id], device=DEVICE, dtype=DTYPE))
        if (length := sum(x.shape[0] for x in tensors)) < L:
            tensors.append(torch.full((L - length,), pad_token_id, device=DEVICE, dtype=DTYPE))

        reversed_input_ids[i] = torch.cat(tensors)

    return reversed_input_ids


def test_prepare_input_for_backward_model():
    pad_token_id = 0
    bos_token_id = 42
    eos_token_id = 100

    print("Test 1\n------")
    input_ids = [[1, 2, 3, 4, pad_token_id], [5, 6, 7, 8, 9]]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    expected_reversed_input_ids = [[4, 3, 2, 1, pad_token_id], [9, 8, 7, 6, 5]]
    reversed_input_ids = prepare_input_for_backward_model(input_ids, pad_token_id).tolist()
    assert reversed_input_ids == expected_reversed_input_ids, f"Got: {reversed_input_ids}\nExpected{expected_reversed_input_ids}"

    print("Test 2\n------")
    input_ids = [[bos_token_id, 1, 2, 3, 4, pad_token_id], [bos_token_id, 5, 6, 7, 8, 9]]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    expected_reversed_input_ids = [[bos_token_id, 4, 3, 2, 1, pad_token_id], [bos_token_id, 9, 8, 7, 6, 5]]
    reversed_input_ids = prepare_input_for_backward_model(input_ids, pad_token_id, bos_token_id).tolist()
    assert reversed_input_ids == expected_reversed_input_ids, f"Got: {reversed_input_ids}\nExpected{expected_reversed_input_ids}"

    print("Test 3\n------")
    input_ids = [[1, 2, 3, 4, eos_token_id, pad_token_id], [5, 6, 7, 8, 9, eos_token_id]]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    expected_reversed_input_ids = [[4, 3, 2, 1, eos_token_id, pad_token_id], [9, 8, 7, 6, 5, eos_token_id]]
    reversed_input_ids = prepare_input_for_backward_model(input_ids, pad_token_id, None, eos_token_id).tolist()
    assert reversed_input_ids == expected_reversed_input_ids, f"Got: {reversed_input_ids}\nExpected{expected_reversed_input_ids}"

    print("Test 4\n------")
    input_ids = [[bos_token_id, 1, 2, 3, 4, eos_token_id, pad_token_id], [bos_token_id, 5, 6, 7, 8, 9, eos_token_id]]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    expected_reversed_input_ids = [[bos_token_id, 4, 3, 2, 1, eos_token_id, pad_token_id], [bos_token_id, 9, 8, 7, 6, 5, eos_token_id,]]
    reversed_input_ids = prepare_input_for_backward_model(input_ids, pad_token_id, bos_token_id, eos_token_id).tolist()
    assert reversed_input_ids == expected_reversed_input_ids, f"Got: {reversed_input_ids}\nExpected{expected_reversed_input_ids}"


class MambaConfig(PretrainedConfig):

    def __init__(
        self,
        d_model: int = 2560,
        n_layer: int = 64,
        vocab_size: int = 50277,
        ssm_cfg: Optional[dict] = None,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 8,
        initializer_range: float = 0.02,
        rescale_prenorm_residual: bool = True,
        n_residuals_per_layer: int = 1,
        norm_epsilon: float = 1e-5,
        pad_token_id: int = -1,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        mlp_hidden_size: int = 512,
        mode: Literal["uni", "bi"] = "uni",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.vocab_size = vocab_size
        self.ssm_cfg = {} if ssm_cfg is None else ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.initializer_range = initializer_range
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.n_residuals_per_layer = n_residuals_per_layer
        self.norm_epsilon = norm_epsilon
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mlp_hidden_size = mlp_hidden_size
        self.mode = mode
        # Not having an attribute `pruned_heads` causes an error somewhere in transformers
        # (TODO: document where). Having pruned_heads=None causes issues when loading from
        # a pre-trained model. The solution is to set pruned_heads to an empty dictionary.
        # TODO: assert that this is true...
        self.pruned_heads = {}


class MambaPreTrainedModel(PreTrainedModel):

    config_class = MambaConfig
    load_tf_weights = None
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    config: MambaConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.n_residuals_per_layer * self.config.n_layer)


class MixerModel(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig) -> None:
        super().__init__(config)

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers: list[Block] = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model,
            eps=config.norm_epsilon,
        )
        self.post_init()

    def allocate_inference_cache(self, batch_size, max_seqlen, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids: LongTensor):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        if not self.config.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.config.residual_in_fp32,
            )
        return hidden_states


class BiMixerModel(MambaPreTrainedModel):

    """
    Implementation of the bidirectional model described in
    "Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling"
    https://arxiv.org/pdf/2403.03234.pdf
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__(config)

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers_forward: list[Block] = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )

        self.layers_backward: list[Block] = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )
        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model,
            eps=config.norm_epsilon,
        )

        # Tie some of the forward and backward weights
        for i in range(len(self.layers_forward)):
            mamba_block_forward: Mamba = self.layers_forward[i].mixer
            mamba_block_backward: Mamba = self.layers_forward[i].mixer

            # Which linear projections should be shared? Options include: in_proj, out_proj, x_proj, dt_proj
            # based on the paper, it seems most probable that they share in_proj, out_proj, x_proj but not
            # dt_proj cause its associated with the SSM

            mamba_block_forward.in_proj = mamba_block_backward.in_proj
            mamba_block_forward.out_proj = mamba_block_backward.out_proj
            mamba_block_forward.x_proj = mamba_block_backward.x_proj

        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

        self.post_init()

    # FIXME: does this work?
    def allocate_inference_cache(self, batch_size, max_seqlen, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, **kwargs)
            for i, layer in enumerate(self.layers_forward + self.layers_backward)
        }

    def forward(self, input_ids: LongTensor, input_ids_backward: Optional[LongTensor] = None):
        input_ids_forward = input_ids
        if input_ids_backward is None:
            input_ids_backward = prepare_input_for_backward_model(
                input_ids_forward,
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                "right",
            )

        hidden_states_forward = self.embedding(input_ids_forward)
        hidden_states_backward = self.embedding(input_ids_backward)

        residual_forward, residual_backward = None, None
        for layer_forward, layer_backward in zip(self.layers_forward, self.layers_backward):
            hidden_states_forward, residual_forward = layer_forward(hidden_states_forward, residual_forward)
            hidden_states_backward, residual_backward = layer_backward(hidden_states_backward, residual_backward)

            hidden_states = hidden_states_forward + hidden_states_backward.flip(1)
            hidden_states_forward = hidden_states
            hidden_states_backward = hidden_states

            if residual_forward is not None and residual_backward is not None:
                residual = residual_forward + residual_backward.flip(1)
                residual_forward = residual
                residual_backward = residual


        if not self.config.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.config.residual_in_fp32,
            )

        return hidden_states


class MambaLMHeadModel(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig) -> None:
        super().__init__(config)
        self.backbone = MixerModel(config)
        self.head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )
        # FIXME: should the weights be tied in another manner?
        self.tie_weights()
        self.post_init()

    def tie_weights(self):
        self.head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, **kwargs)

    def forward(self, input_ids: LongTensor, labels: Optional[LongTensor] = None) -> CausalLMOutput:
        hidden_states: Tensor = self.backbone(input_ids)
        logits: Tensor = self.head(hidden_states)

        loss = None
        labels = input_ids
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # use TrainingArguments.prediction_loss_only to prevent OOMs
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)


class MambaForMaskedLM(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig) -> None:
        super().__init__(config)
        self.backbone = BiMixerModel(config)
        self.head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )
        # FIXME: should the weights be tied in another manner?
        self.tie_weights()
        self.post_init()

    def tie_weights(self):
        self.head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, **kwargs)

    def forward(self, input_ids: LongTensor, labels: Optional[LongTensor] = None) -> CausalLMOutput:
        hidden_states: Tensor = self.backbone(input_ids)
        logits: Tensor = self.head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)


class MambaForSequenceClassification(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig) -> None:
        super().__init__(config)

        if config.mode == "uni":
            self.backbone = MixerModel(config)
        elif config.mode == "bi":
            self.backbone = BiMixerModel(config)
        else:
            raise ValueError(f"Unsupported {config.mode=}")

        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(config.mlp_hidden_size, config.num_labels),
        )

        self.num_labels = config.num_labels
        self.post_init()

    def forward(self, input_ids: LongTensor, labels: Optional[LongTensor] = None) -> CausalLMOutput:
        hidden_states: Tensor = self.backbone(input_ids)
        logits: Tensor = self.head(hidden_states)

        batch_size = input_ids.shape[0]
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        # Returning hidden_states can cause excessive memory build-up during evaluation
        # use TrainingArguments.prediction_loss_only to prevent OOMs is insufficient because
        # we need the logits.
        return SequenceClassifierOutput(loss=loss, logits=pooled_logits, hidden_states=None)


if __name__ == "__main__":
    test_prepare_input_for_backward_model()
