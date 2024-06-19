"""
A huggingface-compatible implementation of MalConv.

TODO:
 - remove some of the comments and assertions...
"""

import math
import os
import sys
from typing import Optional

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput


class MalConvConfig(PretrainedConfig):

    """
    Configuration used by original authors:

        >>> MalConvConfig(
                vocab_size=257,
                embedding_size=8,
                channels=128,
                stride=500,
                kernel_size=500,
                pad_token_id=0,
            )
    """

    def __init__(
        self,
        vocab_size: int = 264,
        embedding_size: int = 256,
        channels: int = 128,
        stride: int = 512,
        kernel_size: int = 512,
        pad_token_id: int = 0,
        **kwds,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.channels = channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.pad_token_id = pad_token_id
        super().__init__(**kwds)


class MalConvPreTrainedModel(PreTrainedModel):

    config_class = MalConvConfig
    base_model_prefix = "malconv"
    supports_gradient_checkpointing = False


class MalConv(MalConvPreTrainedModel):
    """
    Based on Figure 6 in the full-length MalConv paper, accessible here:
        https://www.semanticscholar.org/reader/4417dfcfc722b8b31278a0ebcc1595963dab5a1c
    """

    def __init__(self, config: MalConvConfig):
        super().__init__(config)
        self.config = config

        self.embed = nn.Embedding(
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

    def forward(self, input_ids: Tensor) -> BaseModelOutput:

        # Annotations:
        # B: batch size
        # L: sequence length
        # E: embedding size
        # C: channels
        # S: stride

        B = input_ids.shape[0]
        L = input_ids.shape[1]
        E = self.config.embedding_size
        C = self.config.channels
        S = math.floor((L - self.config.kernel_size) / self.config.stride + 1)

        input_ids: Tensor  # [B, L]
        input_embeddings: Tensor = self.embed(input_ids).transpose(1, 2)  # [B, E, L]
        assert tuple(input_embeddings.shape) == (B, E, L), f"{input_embeddings.shape=} != {(B, E, L)}"

        cnn_1_value: Tensor = self.conv_1(input_embeddings)  # [B, C, S - 1]
        cnn_2_value: Tensor = self.conv_2(input_embeddings)  # [B, C, S - 1]
        assert tuple(cnn_1_value.shape) == (B, C, S), f"{cnn_1_value.shape=} != {(B, C, S)}"
        assert tuple(cnn_2_value.shape) == (B, C, S), f"{cnn_2_value.shape=} != {(B, C, S)}"

        gating_value: Tensor = cnn_1_value * F.sigmoid(cnn_2_value)  # [B, C, S - 1]
        assert tuple(gating_value.shape) == (B, C, S), f"{gating_value.shape=} != {(B, C, S)}"
        pooled_value: Tensor = self.pooling(gating_value)  # [B, C, 1]
        assert tuple(pooled_value.shape) == (B, C, 1), f"{pooled_value.shape=} != {(B, C, 1)}"

        hidden_states: Tensor = pooled_value.squeeze(-1)  # [B, C]
        assert tuple(hidden_states.shape) == (B, C), f"{hidden_states.shape=} != {(B, C)}"

        return BaseModelOutput(hidden_states)


class MalConvForSequenceClassification(MalConvPreTrainedModel):

    def __init__(self, config: MalConvConfig):
        super().__init__(config)
        self.malconv = MalConv(config)
        self.clf_head = nn.Linear(config.channels, config.num_labels)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> SequenceClassifierOutput:
        x: Tensor = self.malconv(input_ids)[0]
        x: Tensor = self.clf_head(x)

        logits = x
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MalConvConfig(
        vocab_size=256,
        embedding_size=256,
        channels=128,
        stride=256,
        kernel_size=512,
        pad_token_id=0,
    )
    model = MalConvForSequenceClassification(config).to(device)

    length = 2 ** 19 + 1
    bos = torch.tensor([1])
    eos = torch.tensor([2])
    x = torch.randint(3, config.vocab_size, (length - 2,))
    x = torch.cat([bos, x, eos], dim=0)
    x = x.unsqueeze(0).to(device)

    model(x)


if __name__ == "__main__":
    test()
