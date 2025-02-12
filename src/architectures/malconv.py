"""
MalConv.
"""

from typing import Optional

import math
from torch import nn, Tensor, IntTensor, FloatTensor
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput

from src.architectures.head_utils import Head, pool_logits, get_clf_loss


class MalConvConfig(PretrainedConfig):
    """
    Configuration used by original authors:

        >>> MalConvConfig(
                vocab_size=257,
                embedding_size=8,
                channels=128,
                stride=500,
                kernel_size=500,
                mlp_hidden_size=128,
                pad_token_id=0,
            )
    """

    def __init__(
        self,
        vocab_size: int = 264,
        embedding_size: int = 256,
        pad_token_id: int = 0,
        channels: int = 128,
        stride: int = 512,
        kernel_size: int = 512,
        head_hidden_size: int = 0,
        head_num_hidden_layers: int = 0,
        head_dropout: float = 0.1,
        **kwds,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pad_token_id = pad_token_id
        self.channels = channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.head_hidden_size = head_hidden_size
        self.head_num_hidden_layers = head_num_hidden_layers
        self.head_dropout = head_dropout
        super().__init__(**kwds)

    @property
    def hidden_size(self) -> int:
        return self.channels


class MalConvPreTrainedModel(PreTrainedModel):
    config_class = MalConvConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = False
    config: MalConvConfig


class MalConv(MalConvPreTrainedModel):

    def __init__(self, config: MalConvConfig):

        super().__init__(config)

        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            padding_idx=config.pad_token_id,
        )

        self.conv_1 = nn.Conv1d(
            in_channels=config.embedding_size,
            out_channels=config.channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
        )

        self.conv_2 = nn.Conv1d(
            in_channels=config.embedding_size,
            out_channels=config.channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
        )

        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, input_ids: Optional[IntTensor] = None, inputs_embeds: Optional[FloatTensor] = None) -> BaseModelOutput:

        # B: batch size
        # L: sequence length
        # E: embedding size
        # C: channels
        # S: stride

        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        L = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        E = self.config.embedding_size
        C = self.config.channels
        S = math.floor((L - self.config.kernel_size) / self.config.stride + 1)

        if input_ids is not None:
            assert tuple(input_ids.shape) == (B, L), f"{input_ids.shape} != {(B, L)}"

        inputs_embeds = inputs_embeds if inputs_embeds is not None else self.embeddings(input_ids)
        inputs_embeds = inputs_embeds.transpose(1, 2)
        assert tuple(inputs_embeds.shape) == (B, E, L), f"{inputs_embeds.shape} != {(B, E, L)}"

        cnn_1_value: Tensor = self.conv_1(inputs_embeds)
        assert tuple(cnn_1_value.shape) == (B, C, S), f"{cnn_1_value.shape} != {(B, C, S)}"

        cnn_2_value: Tensor = self.conv_2(inputs_embeds)
        assert tuple(cnn_2_value.shape) == (B, C, S), f"{cnn_2_value.shape} != {(B, C, S)}"

        gating_value: Tensor = cnn_1_value * F.sigmoid(cnn_2_value)
        assert tuple(gating_value.shape) == (B, C, S), f"{gating_value.shape} != {(B, C, S)}"

        pooled_value: Tensor = self.pooling(gating_value)
        assert tuple(pooled_value.shape) == (B, C, 1), f"{pooled_value.shape} != {(B, C, 1)}"

        hidden_states: Tensor = pooled_value.squeeze(-1)
        assert tuple(hidden_states.shape) == (B, C), f"{hidden_states.shape} != {(B, C)}"

        return BaseModelOutput(hidden_states)


class MalConvForSequenceClassification(MalConvPreTrainedModel):

    def __init__(self, config: MalConvConfig):
        super().__init__(config)
        self.backbone = MalConv(config)
        self.clf_head = Head(
            config.hidden_size,
            config.num_labels,
            config.head_hidden_size,
            config.head_num_hidden_layers,
            config.head_dropout,
        )

    def forward(self, input_ids: Optional[IntTensor] = None, inputs_embeds: Optional[FloatTensor] = None, labels: Optional[Tensor] = None) -> SequenceClassifierOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        hidden_states = self.backbone.forward(input_ids=input_ids, inputs_embeds=inputs_embeds)[0]
        logits = self.clf_head.forward(hidden_states.unsqueeze(1)).squeeze(1)
        loss = get_clf_loss(logits, labels, self.config.num_labels, self.config.problem_type) if labels is not None else None
        return SequenceClassifierOutput(loss, logits, hidden_states)
