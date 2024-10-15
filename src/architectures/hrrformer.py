"""
Implementation of HRRFormer.
"""

import warnings
from typing import Literal, Optional

import torch
import torch.utils.checkpoint
from torch import nn
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from src.utils import log_tensor
from src.architectures.utils import binding, unbinding, cosine_similarity
from src.architectures.head_utils import Head, pool_logits, get_clf_loss, get_clm_loss, get_mlm_loss


ARG_REQUIRED = -1
ARG_INFERRED = -1


class HRRConfig(PretrainedConfig):

    def __init__(
        self,
        vocab_size: int = ARG_REQUIRED,
        hidden_size: int = ARG_REQUIRED,
        num_hidden_layers: int = ARG_REQUIRED,
        num_attention_heads: int = ARG_REQUIRED,
        embedding_size: int = ARG_INFERRED,
        intermediate_size: int = ARG_INFERRED,
        head_hidden_size: int = 0,
        head_num_hidden_layers: int = 0,
        head_dropout: float = 0.1,
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
        fft_norm: Literal["forward", "backward", "ortho"] = "backward",
        **kwargs,
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.embedding_size = hidden_size if embedding_size == ARG_INFERRED else embedding_size
        self.intermediate_size = hidden_size * 4 if intermediate_size == ARG_INFERRED else intermediate_size

        self.head_hidden_size = head_hidden_size
        self.head_num_hidden_layers = head_num_hidden_layers
        self.head_dropout = head_dropout
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.fft_norm = fft_norm

        super().__init__(pad_token_id=pad_token_id, **kwargs)

class HRRFormerEmbeddings(BertEmbeddings):

    def __init__(self, config: HRRConfig) -> None:
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)


class HRROutput(BertOutput):
    ...


class HRRSelfOutput(BertSelfOutput):
    ...


class HRRIntermediate(BertIntermediate):
    ...


del BertEmbeddings
del BertSelfOutput
del BertOutput
del BertIntermediate


class HRRSelfAttention(nn.Module):

    def __init__(self, config: HRRConfig, position_embedding_type: Optional[str] = None) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
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
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.fft_norm = config.fft_norm

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[Tensor]:
        if attention_mask is not None:
            if attention_mask.shape[1] != 1 or attention_mask.shape[2] != 1:
                raise ValueError(
                    f"Recieved an attention_mask with shape: {attention_mask.shape}."
                    "This is a causal attention mask, which is not needed for HRRFormer and can cause OOM."
                    "Pass in an attention_mask with shape (B, 1, 1, T) instead to mask out specific tokens, e.g., padding."
                )
        if encoder_attention_mask is not None:
            if encoder_attention_mask.shape[1] != 1 or encoder_attention_mask.shape[2] != 1:
                raise ValueError(
                    f"Recieved an encoder_attention_mask with shape: {encoder_attention_mask.shape}."
                    "This is a causal attention mask, which is not needed for HRRFormer and can cause OOM."
                    "Pass in an attention_mask with shape (B, 1, 1, T) instead to mask out specific tokens, e.g., padding."
                )

        # When used as a decoder, a causal mask is expected. This mask is not
        # applied to the attention scores as would usually be done, but rather
        # to the superpositional binding of the key and value tensors.

        # pylint: disable=unused-variable
        B = hidden_states.size(0)     # - batch size
        H = self.num_attention_heads  # - number of attention heads
        T = hidden_states.size(1)     # - sequence length
        D = self.attention_head_size  # - effective hidden size
        # pylint: enable=unused-variable

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        if is_cross_attention and past_key_value is not None:
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

        key_layer: Tensor    # (B, H, T, D)
        value_layer: Tensor  # (B, H, T, D)
        query_layer: Tensor  # (B, H, T, D)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        pad_to = 1 << (D - 1).bit_length()

        # log(key_layer, "key_layer")
        # log(value_layer, "value_layer")
        # log(query_layer, "query_layer")

        # Binding and unbinding
        superpositions = binding(key_layer, value_layer, dim=-1, norm=self.fft_norm, n=pad_to)[:,:,:,0:D]      # (B, H, T, D)
        if self.is_decoder and attention_mask is not None:
            # Causal masking needs to take place within the superposition.
            # We create T superpositions using interactions from the preceeding tokens.
            # This ensures that the `superposition` and `value_approx` preserves causality.
            superposition = torch.cumsum(superpositions, dim=-2)                                               # (B, H, T, D)
        else:
            superposition = torch.sum(superpositions, dim=-2, keepdims=True)                                   # (B, H, 1, D)
        value_approx = unbinding(superposition, query_layer, dim=-1, norm=self.fft_norm, n=pad_to)[:,:,:,0:D]  # (B, H, T, D)
        attention_scores = cosine_similarity(value_layer, value_approx, dim=-1, keepdim=True)                  # (B, H, T, 1)

        # log(superpositions, "superpositions")
        # log(superposition, "superposition")
        # log(value_approx, "value_approx")
        # log(attention_scores, "attention_scores")

        # Attention mask, scores, and probabilities
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.permute(0, 1, 3, 2)      # (B, H, T, 1)
        attention_probs = F.softmax(attention_scores, dim=-2)                             # (B, H, T, 1)
        attention_probs = self.dropout.forward(attention_probs)                           # (B, H, T, 1)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask                                 # (B, H, T, 1)

        # Attention module context
        context_layer = attention_probs * value_layer                                         # (B, H, T, D)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()                        # (B, T, H, D)
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,)) # (B, T, H * D)

        # Determine outputs and return
        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (attention_probs,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class HRRPreTrainedModel(PreTrainedModel):

    config_class = HRRConfig
    base_model_prefix = "backbone"
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

    def __init__(self, config: HRRConfig, position_embedding_type: Optional[str] = None) -> None:
        super().__init__()
        self.self = HRRSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = HRRSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return

        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
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
        hidden_states: Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[Tensor]:
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
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class HRRLayer(nn.Module):

    def __init__(self, config: HRRConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = HRRAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = HRRAttention(config, position_embedding_type="absolute")
        self.intermediate = HRRIntermediate(config)
        self.output = HRROutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[Tensor]:
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

        if self.is_decoder:  # if decoder, the last output is tuple of self-attn cache
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:                # add self attentions if we output attention weights
            outputs = self_attention_outputs[1:]

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
            outputs = outputs + cross_attention_outputs[1:-1] # add cross attentions if we output attention weights

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

    def feed_forward_chunk(self, attention_output: Tensor) -> Tensor:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class HRREncoder(nn.Module):

    def __init__(self, config: HRRConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([HRRLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                warnings.warn("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
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

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class HRRModel(HRRPreTrainedModel):

    def __init__(self, config: HRRConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embeddings = HRRFormerEmbeddings(config)
        self.embedding_projection = nn.Linear(config.embedding_size, config.hidden_size) if config.embedding_size != config.hidden_size else nn.Identity()
        self.encoder = HRREncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        use_cache = False
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify input_ids and inputs_embeds.")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if token_type_ids.dtype != torch.long:
            warnings.warn(f"{token_type_ids.dtype=}, which is unexpected. Casting to torch.long")
            token_type_ids = token_type_ids.to(torch.long)

        # NOTE: The causal implementation of HRRFormer does not use the attention mask.
        # The causal attention mask is memory hungry, so we just ignore it.
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        extended_attention_mask = attention_mask[:, None, None, :]

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

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_output = self.embedding_projection(embedding_output)

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
        )
        sequence_output = encoder_outputs[0]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class HRRForCausalLM(HRRPreTrainedModel):

    _tied_weights_keys = ["head_clm.weight_to_tie"]

    def __init__(self, config: HRRConfig) -> None:
        if not config.is_decoder:
            raise ValueError(f"{config.is_decoder=} (expected True)")
        super().__init__(config)
        self.config: HRRConfig
        self.backbone = HRRModel(config)
        self.head_clm = Head(
            config.hidden_size,
            config.vocab_size,
            config.head_hidden_size,
            config.head_num_hidden_layers,
            config.head_dropout,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        return self.backbone.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Linear:
        return self.head_clm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.head_clm.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[list[Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:

        if labels is not None:
            use_cache = False

        outputs = self.backbone.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.head_clm(outputs.last_hidden_state)
        logits = pool_logits("none", logits, input_ids, self.config.pad_token_id)
        loss = get_clm_loss(logits, labels, self.config.vocab_size) if labels is not None else None

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class HRRForMaskedLM(HRRPreTrainedModel):

    _tied_weights_keys = ["head_mlm.weight_to_tie"]

    def __init__(self, config: HRRConfig):
        if config.is_decoder:
            raise ValueError(f"{config.is_decoder=} (expected False)")
        super().__init__(config)
        self.config: HRRConfig
        self.backbone = HRRModel(config)
        self.head_mlm = Head(
            config.hidden_size,
            config.vocab_size,
            config.head_hidden_size,
            config.head_num_hidden_layers,
            config.head_dropout,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        return self.backbone.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Linear:
        return self.head_mlm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.head_mlm.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> MaskedLMOutput:

        outputs = self.backbone.forward(
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
        )
        logits = self.head_mlm.forward(outputs.last_hidden_state)
        logits = pool_logits("none", logits, input_ids, self.config.pad_token_id)
        loss = get_mlm_loss(logits, labels, self.config.vocab_size) if labels is not None else None

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HRRForSequenceClassification(HRRPreTrainedModel):

    def __init__(self, config: HRRConfig):
        super().__init__(config)
        self.config: HRRConfig
        self.backbone = HRRModel(config)
        self.head_clf = Head(
            config.hidden_size,
            config.num_labels,
            config.head_hidden_size,
            config.head_num_hidden_layers,
            config.head_dropout,
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutput:

        outputs = self.backbone.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.head_clf.forward(outputs.last_hidden_state)
        logits = pool_logits("last" if self.config.is_decoder else "mean", logits, input_ids, self.config.pad_token_id)
        loss = get_clf_loss(logits, labels, self.config.num_labels, self.config.problem_type) if labels is not None else None

        # Returning hidden_states can cause excessive memory build-up during evaluation.
        # Using TrainingArguments.prediction_loss_only is insufficient because we need the logits.
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
