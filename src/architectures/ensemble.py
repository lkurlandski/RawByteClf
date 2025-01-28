"""
Utilities for ensembling multiple models together.
"""

from typing import Callable, Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

from src.architectures.head_utils import Head, get_clf_loss


class EnsembleForSequenceClassification(PreTrainedModel):

    def __init__(self, config: PretrainedConfig, backbone: Callable[[PretrainedConfig], PreTrainedModel]) -> None:
        super().__init__(config)
        self.config = config
        self.raw_backbone = backbone(config)
        self.dis_backbone = backbone(config)
        self.dec_backbone = backbone(config)
        self.head_clf = Head(
            config.hidden_size * 3,
            config.num_labels,
            config.head_hidden_size,
            config.head_num_hidden_layers,
            config.head_dropout,
        )
        self.post_init()

    def forward(
        self,
        raw_input_ids: Optional[Tensor] = None,
        dis_input_ids: Optional[Tensor] = None,
        dec_input_ids: Optional[Tensor] = None,
        raw_attention_mask: Optional[Tensor] = None,
        dis_attention_mask: Optional[Tensor] = None,
        dec_attention_mask: Optional[Tensor] = None,
        raw_token_type_ids: Optional[Tensor] = None,
        dis_token_type_ids: Optional[Tensor] = None,
        dec_token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> SequenceClassifierOutput:

        raw_hidden_states = self.get_pooled_hidden_states(self.raw_backbone, input_ids=raw_input_ids,
            attention_mask=raw_attention_mask, token_type_ids=raw_token_type_ids)
        dis_hidden_states = self.get_pooled_hidden_states(self.dis_backbone, input_ids=dis_input_ids,
            attention_mask=dis_attention_mask, token_type_ids=dis_token_type_ids)
        dec_hidden_states = self.get_pooled_hidden_states(self.dec_backbone, input_ids=dec_input_ids,
            attention_mask=dec_attention_mask, token_type_ids=dec_token_type_ids)

        hidden_states = torch.cat([raw_hidden_states, dis_hidden_states, dec_hidden_states], dim=1).unsqueeze(1)
        logits = self.get_logits(hidden_states)
        logits = logits.squeeze(1)
        loss = get_clf_loss(logits, labels, self.config.num_labels, self.config.problem_type) if labels is not None else None

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def get_pooled_hidden_states(self, backbone: PreTrainedModel, **kwds) -> Tensor:
        return backbone.forward(**kwds).last_hidden_state

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        return self.head_clf.forward(hidden_states)
