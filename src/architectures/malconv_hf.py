"""
A huggingface-compatible implementation of MalConv.
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

import os
import sys
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput


class MalConvConfig(PretrainedConfig):

    def __init__(
        self,
        vocab_size: int = 257,
        embedding_size: int = 8,
        pad_token_id: int = 0,
        window_size: int = 512,
        channels: int = 128,
        stride: int = 512,
        **kwds
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pad_token_id = pad_token_id
        self.window_size = window_size
        self.channels = channels
        self.stride = stride
        super().__init__(**kwds)


class MalConvPreTrainedModel(PreTrainedModel):

    config_class = MalConvConfig
    base_model_prefix = "malconv"
    supports_gradient_checkpointing = False

    # def _init_weights(self, module):
    #     """Initialize the weights"""
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()


class MalConv(MalConvPreTrainedModel):
    """
    Adapted from:
        https://github.com/Alexander-H-Liu/MalConv-Pytorch/blob/master/src/model.py
        https://github.com/elastic/ember/blob/master/malconv/malconv.py
        https://github.com/lkurlandski/MalConv2/blob/main/LowMemConv.py
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
            int(config.embedding_size / 2),
            config.channels,
            config.window_size,
            stride=config.window_size,
            bias=True,
        )
        self.conv_2 = nn.Conv1d(
            int(config.embedding_size / 2),
            config.channels,
            config.window_size,
            stride=config.window_size,
            bias=True,
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> BaseModelOutput:
        x = input_ids
        if x.dim() not in (1, 2):
            raise ValueError(f"Expected 1D or 2D input, got {x.dim()}D input.")
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] < 3:
            raise ValueError("Expecting input of length at least 3 [BOS, ..., EOS]")
        if x.shape[1] < self.config.window_size:
            raise ValueError(f"Expecting input of length at least {self.config.window_size=}")

        x: Tensor = self.embed(input_ids)
        x: Tensor = torch.transpose(x, -1, -2)

        cnn_1_input: Tensor = x.narrow(-2, 0, 4)
        cnn_1_value: Tensor = self.conv_1(cnn_1_input)

        cnn_2_input = x.narrow(-2, 4, 4)
        cnn_2_value = self.conv_2(cnn_2_input)
        gating_weight: Tensor = F.sigmoid(cnn_2_value)

        x: Tensor = cnn_1_value * gating_weight
        x: Tensor = self.pooling(x)
        x: Tensor = x.view(-1, self.config.channels)

        return BaseModelOutput(last_hidden_state=x)


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
        x: Tensor = self.malconv(input_ids, labels=labels)[0]
        x: Tensor = self.clf_head(x)

        logits = x
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
