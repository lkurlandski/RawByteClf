"""PyTorch BERT model."""

import math
import os
from pathlib import Path
import sys
import warnings
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertPreTrainingHeads,
    BertForPreTrainingOutput,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from src.utils import log_tensor
from src.architectures.utils import binding, unbinding, cosine_similarity

# from HRR.with_pytorch import binding, unbinding, cosine_similarity


__all__ = [
    "HRRConfig",
    "HRRSelfAttention",
    "HRRAttention",
    "HRRLayer",
    "HRREncoder",
    "HRRModel",
    "HRRForMaskedLM",
    "HRRForSequenceClassification",
]


logger = logging.get_logger(__name__)


class HRRConfig(PretrainedConfig):
    """
    This is a copy-paste of the BertConfig class with one additional parameter.
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
        superposition_scale_factor: Optional[float | Literal["max", "mean", "log", "norm"]] = None,
        attention_score_scale_factor: Optional[float] = None,
        tensor_log_path: Optional[str] = None,
        tensor_logging: bool = False,
        norm: Literal["forward", "backward", "norm"] = "norm",
        **kwargs,
    ):
        """
        Args
          superposition_scale_factor: A scaling factor to apply to the superpositions before
            summation. This can be a scalar floating point value, `max`, which will scale by the
            square root of the maximum possible value for the sequence length for this model,
            `mean`, which will scale by the length of the input sequence, or `log`, which perform
            the summation in a more stable log space. If None, defaults to 1.0.
          attention_score_scale_factor: A scaling factor to apply to the attention scores before.
            If None, defaults to 1 / sqrt(attention_head_size), as done in the original Transformer.
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.norm = norm
        self.tensor_log_path = None
        if tensor_logging:
            self.tensor_log_path = tensor_log_path.as_posix() if isinstance(tensor_log_path, Path) else tensor_log_path

        if isinstance(superposition_scale_factor, (float, int)):
            self.superposition_scale_factor = float(superposition_scale_factor)
        elif superposition_scale_factor == "max":
            self.superposition_scale_factor = 1.0 / math.sqrt(max_position_embeddings)
        elif superposition_scale_factor in ("mean", "log", "norm"):
            self.superposition_scale_factor = superposition_scale_factor
        elif superposition_scale_factor is None:
            self.superposition_scale_factor = 1.0
        else:
            raise ValueError(f"Invalid value {superposition_scale_factor=}")

        if isinstance(attention_score_scale_factor, (float, int)):
            self.attention_score_scale_factor = float(attention_score_scale_factor)
        elif attention_score_scale_factor is None:
            self.attention_score_scale_factor = 1 / math.sqrt(int(self.hidden_size / self.num_attention_heads))
        else:
            raise ValueError(f"Invalid value {attention_score_scale_factor=}")


class HRRSelfAttention(nn.Module):
    def __init__(self, config: HRRConfig, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

        self.superposition_scale_factor = config.superposition_scale_factor
        self.attention_score_scale_factor = config.attention_score_scale_factor
        self.tensor_log_path = config.tensor_log_path
        self.norm = config.norm

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.tensor_log_path:
            log_tensor(self.tensor_log_path, key_layer, "key_layer")
            log_tensor(self.tensor_log_path, value_layer, "value_layer")
            log_tensor(self.tensor_log_path, query_layer, "query_layer")

        # HRR
        # H' = hidden_size / num_attention_heads  RuntimeError: cuFFT error: CUFFT_INVALID_SIZE
        try:
            superpositions = binding(
                key_layer,
                value_layer,
                dim=-1,
                norm=self.norm,
            )  # (B, h, T, H')
        except RuntimeError:  # TODO: hyperparameter tuning can sometimes induce this error...why?
            print(f"{key_layer.shape=}")
            print(f"{value_layer.shape=}")
            raise

        # Calculating the superposition as the sum of the superpositions can lead to overflow,
        # especially when using fp16. After some analysis, we find that aggressive gradient
        # clipping can alleviate this problem to some degree. However, we also found that scaling
        # the superpositions by a constant factor, e.g., 1 / sqrt(H'), or taking the mean instead
        # of the sum does not seem to impact learning and can also be used to avoid overflow.
        # so we use the mean. torch.mean() also causes overflow, so we compute it manually instead.
        superposition: torch.Tensor  # (B, h, 1, H')
        if self.superposition_scale_factor == "mean":
            superposition = torch.sum(torch.div(superpositions, superpositions.size(-2)), dim=-2, keepdim=True)
        elif self.superposition_scale_factor == "log":
            max_value, _ = torch.max(superpositions, dim=-2, keepdim=True)
            superposition = torch.log(torch.sum(torch.exp(superpositions - max_value), dim=-2, keepdim=True)) + max_value
        elif self.superposition_scale_factor == "norm":
            superpositions = F.normalize(superpositions, p=2.0, dim=-2, eps=1e-12, out=None)
            superposition = torch.sum(superpositions, dim=-2, keepdims=True)
        else:
            superposition = torch.sum(superpositions * self.superposition_scale_factor, dim=-2, keepdims=True)

        value_approx = unbinding(
            superposition,
            query_layer,
            dim=-1,
            norm=self.norm,
        )  # (B, h, T, H')
        attention_scores = cosine_similarity(value_layer, value_approx, dim=-1, keepdim=True)  # (B, h, T, 1)

        if self.tensor_log_path:
            log_tensor(self.tensor_log_path, superpositions, "superpositions")
            log_tensor(self.tensor_log_path, superposition, "superposition")
            log_tensor(self.tensor_log_path, value_approx, "value_approx")
            log_tensor(self.tensor_log_path, attention_scores, "attention_scores")

        # Add the positional encoding
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise NotImplementedError()
            # query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            # if use_cache:
            #     position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            # else:
            #     position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            # position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            # distance = position_ids_l - position_ids_r

            # positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            # positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            # if self.position_embedding_type == "relative_key":
            #     relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            #     attention_scores = attention_scores + relative_position_scores
            # elif self.position_embedding_type == "relative_key_query":
            #     relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            #     relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
            #     attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores * self.attention_score_scale_factor
        # Apply the attention mask.
        # TODO: for causal language modeling, the attention mask has shape (B, 1, T, T).
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.permute(0, 1, 3, 2)  # (B, h, T, 1)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-2)
        if self.tensor_log_path:
            log_tensor(self.tensor_log_path, attention_probs, "attention_probs")
        attention_probs = self.dropout(attention_probs)
        if self.tensor_log_path:
            log_tensor(self.tensor_log_path, attention_probs, "attention_probs_post_dropout")

        # Mask heads if we want to
        if head_mask is not None:
            raise NotImplementedError()
            # attention_probs = attention_probs * head_mask

        context_layer: torch.Tensor = attention_probs * value_layer  # (B, h, T, H')
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (B, T, h, H')
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)  # (B, T, h * H')

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            raise NotImplementedError()
            # outputs = outputs + (past_key_value,)

        return outputs


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    raise NotImplementedError()


class HRRPreTrainedModel(PreTrainedModel):

    config_class = HRRConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"  # This is the name of the main nn.Module in the model. Very dumb.
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class HRRAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = HRRSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class HRRLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = HRRAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = HRRAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class HRREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([HRRLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class HRRModel(HRRPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = HRREncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # TODO: verify that these are of dtype long
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if token_type_ids.dtype != torch.long:
            warnings.warn(f"{token_type_ids.dtype=}, which is unexpected. Casting to torch.long")
            token_type_ids = token_type_ids.to(torch.long)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: sometimes token_type_ids has dtype int32, which raises exceptions.
        # if hasattr(input_ids, "dtype"):
        #     print(f"{input_ids.dtype=}")
        # if hasattr(position_ids, "dtype"):
        #     print(f"{position_ids.dtype=}")
        # if hasattr(token_type_ids, "dtype"):
        #     print(f"{token_type_ids.dtype=}")
        # if hasattr(inputs_embeds, "dtype"):
        #     print(f"{inputs_embeds.dtype=}")
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class HRRForMaskedLM(HRRPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = HRRModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class HRRForSequenceClassification(HRRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = HRRModel(config)
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

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


# class HRRLMHeadModel(HRRPreTrainedModel):
#     _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

#     def __init__(self, config):
#         super().__init__(config)

#         if not config.is_decoder:
#             logger.warning(
#                 "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`"
#             )

#         self.bert = HRRModel(config, add_pooling_layer=False)
#         self.cls = BertOnlyMLMHead(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_output_embeddings(self):
#         return self.cls.predictions.decoder

#     def set_output_embeddings(self, new_embeddings):
#         self.cls.predictions.decoder = new_embeddings

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         past_key_values: Optional[List[torch.Tensor]] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         if labels is not None:
#             use_cache = False

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]
#         prediction_scores = self.cls(sequence_output)

#         lm_loss = None
#         if labels is not None:
#             # we are doing next-token prediction; shift prediction scores and input ids by one
#             shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
#             labels = labels[:, 1:].contiguous()
#             loss_fct = CrossEntropyLoss()
#             lm_loss = loss_fct(
#                 shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
#             )

#         if not return_dict:
#             output = (prediction_scores,) + outputs[2:]
#             return ((lm_loss,) + output) if lm_loss is not None else output

#         return CausalLMOutputWithCrossAttentions(
#             loss=lm_loss,
#             logits=prediction_scores,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             cross_attentions=outputs.cross_attentions,
#         )

#     def prepare_inputs_for_generation(
#         self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
#     ):
#         input_shape = input_ids.shape
#         # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
#         if attention_mask is None:
#             attention_mask = input_ids.new_ones(input_shape)

#         # cut decoder_input_ids if past_key_values is used
#         if past_key_values is not None:
#             past_length = past_key_values[0][0].shape[2]

#             # Some generation methods already pass only the last input ID
#             if input_ids.shape[1] > past_length:
#                 remove_prefix_length = past_length
#             else:
#                 # Default to old behavior: keep only final ID
#                 remove_prefix_length = input_ids.shape[1] - 1

#             input_ids = input_ids[:, remove_prefix_length:]

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "past_key_values": past_key_values,
#             "use_cache": use_cache,
#         }

#     def _reorder_cache(self, past_key_values, beam_idx):
#         reordered_past = ()
#         for layer_past in past_key_values:
#             reordered_past += (
#                 tuple(
#                     past_state.index_select(0, beam_idx.to(past_state.device))
#                     for past_state in layer_past
#                 ),
#             )
#         return reordered_past


# class HRRForNextSentencePrediction(HRRPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = HRRModel(config)
#         self.cls = BertOnlyNSPHead(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:

#         if "next_sentence_label" in kwargs:
#             warnings.warn(
#                 "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
#                 " `labels` instead.",
#                 FutureWarning,
#             )
#             labels = kwargs.pop("next_sentence_label")

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs[1]

#         seq_relationship_scores = self.cls(pooled_output)

#         next_sentence_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

#         if not return_dict:
#             output = (seq_relationship_scores,) + outputs[2:]
#             return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

#         return NextSentencePredictorOutput(
#             loss=next_sentence_loss,
#             logits=seq_relationship_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# class HRRForPreTraining(HRRPreTrainedModel):
#     _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = HRRModel(config)
#         self.cls = BertPreTrainingHeads(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_output_embeddings(self):
#         return self.cls.predictions.decoder

#     def set_output_embeddings(self, new_embeddings):
#         self.cls.predictions.decoder = new_embeddings

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         next_sentence_label: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output, pooled_output = outputs[:2]
#         prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

#         total_loss = None
#         if labels is not None and next_sentence_label is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(
#                 prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
#             )
#             next_sentence_loss = loss_fct(
#                 seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
#             )
#             total_loss = masked_lm_loss + next_sentence_loss

#         if not return_dict:
#             output = (prediction_scores, seq_relationship_score) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output

#         return BertForPreTrainingOutput(
#             loss=total_loss,
#             prediction_logits=prediction_scores,
#             seq_relationship_logits=seq_relationship_score,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
