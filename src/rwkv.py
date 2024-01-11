"""

"""

from dataclasses import dataclass
from typing import Optional

from transformers import PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.rwkv.modeling_rwkv import (
    RwkvPreTrainedModel,
    RwkvModel,
)
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss


class RwkvConfig(PretrainedConfig):
    model_type = "rwkv"
    attribute_map = {"max_position_embeddings": "context_length"}

    def __init__(
        self,
        vocab_size=50277,
        context_length=1024,
        hidden_size=4096,
        num_hidden_layers=32,
        attention_hidden_size=None,
        intermediate_size=None,
        layer_norm_epsilon=1e-5,
        bos_token_id=0,
        eos_token_id=0,
        rescale_every=6,
        tie_word_embeddings=False,
        use_cache=True,
        classifier_dropout: Optional[float] = None,
        hidden_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_hidden_size = (
            attention_hidden_size if attention_hidden_size is not None else hidden_size
        )
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else 4 * hidden_size
        )
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rescale_every = rescale_every
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.classifier_dropout = classifier_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class RwkvForSequenceClassification(RwkvPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.rwkv = RwkvModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[list[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskedLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0].mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RwkvForMaskedLM(RwkvPreTrainedModel):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config: RwkvConfig):
        raise NotImplementedError()
        super().__init__(config)
        self.rwkv = RwkvModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[list[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskedLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        logits = self.cls(hidden_states)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
