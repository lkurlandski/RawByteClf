"""
An abominable class for ensembling multiple models together.
"""

from __future__ import annotations
from typing import Callable, Optional, Protocol

import torch
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

from src.architectures.head_utils import Head, get_clf_loss


class BackboneInit(Protocol):

    def __call__(self, config: PretrainedConfig) -> PreTrainedModel:
        ...

    def from_pretrained(self, pretrained_model_name_or_path: str, *args, **kwds) -> PreTrainedModel:
        ...


class EnsembleForSequenceClassification(PreTrainedModel):

    # Class variables
    supports_gradient_checkpointing = False
    backbone_forward_kwds = ("input_ids", "labels", "attention_mask", "token_type_ids")

    # Instance variables
    config: PretrainedConfig
    backbone_init: BackboneInit
    raw_backbone: PreTrainedModel
    dis_backbone: PreTrainedModel
    dec_backbone: PreTrainedModel
    head_clf: Head

    def __init__(self, config: PretrainedConfig, backbone_init: BackboneInit) -> None:
        super().__init__(config)
        self.config = config
        self.backbone_init = backbone_init
        self.raw_backbone = backbone_init(config)
        self.dis_backbone = backbone_init(config)
        self.dec_backbone = backbone_init(config)
        self.head_clf = Head(
            config.hidden_size * 3,
            config.num_labels,
            config.head_hidden_size,
            config.head_num_hidden_layers,
            config.head_dropout,
        )
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | tuple[str] | dict[str],
        backbone_init: BackboneInit,
        *args,
        config: Optional[PretrainedConfig] = None,
        **kwds,
    ) -> EnsembleForSequenceClassification:
        # There appear to be some problems here, as indicated by pylint, but it works when called from a subclass.
        if isinstance(pretrained_model_name_or_path, (tuple, dict)):
            if isinstance(pretrained_model_name_or_path, tuple):
                pretrained_model_name_or_path = {
                    "raw": pretrained_model_name_or_path[0],
                    "dis": pretrained_model_name_or_path[1],
                    "dec": pretrained_model_name_or_path[2],
                }
            for k in ("raw", "dis", "dec"):
                if k not in pretrained_model_name_or_path:
                    raise KeyError(f"Missing key '{k}' in 'pretrained_model_name_or_path'.")

            obj = cls(config)  # pylint: disable=no-value-for-parameter
            obj.raw_backbone = backbone_init.from_pretrained(pretrained_model_name_or_path["raw"], *args, config=config, **kwds)
            obj.dis_backbone = backbone_init.from_pretrained(pretrained_model_name_or_path["dis"], *args, config=config, **kwds)
            obj.dec_backbone = backbone_init.from_pretrained(pretrained_model_name_or_path["dec"], *args, config=config, **kwds)
            return obj

        return super().from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwds)  # pylint: disable=no-member

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
        kwds = {k: v for k, v in kwds.items() if k in self.backbone_forward_kwds}
        return backbone.forward(**kwds).last_hidden_state

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        return self.head_clf.forward(hidden_states)
