import torch
from transformers.models.pegasus.modeling_pegasus import *
from typing import Union, Tuple

import torch.nn.functional as F

_CHECKPOINT_FOR_DOC = "google/pegasus-large"
_CONFIG_FOR_DOC = "PegasusConfig"
_TOKENIZER_FOR_DOC = "PegasusTokenizer"

#
# class PegasusAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""
#
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#         is_decoder: bool = False,
#         bias: bool = True,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.head_dim = embed_dim // num_heads
#
#         if (self.head_dim * num_heads) != self.embed_dim:
#             raise ValueError(
#                 f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
#                 f" and `num_heads`: {num_heads})."
#             )
#         self.scaling = self.head_dim**-0.5
#         self.is_decoder = is_decoder
#
#         self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#
#     def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
#         return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
#
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         key_value_states: Optional[torch.Tensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         layer_head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         """Input shape: Batch x Time x Channel"""
#
#         # if key_value_states are provided this layer is used as a cross-attention layer
#         # for the decoder
#         is_cross_attention = key_value_states is not None
#
#         bsz, tgt_len, _ = hidden_states.size()
#
#         ##
#         bsz = bsz*2
#         ##
#
#         # get query proj
#         query_states = self.q_proj(hidden_states) * self.scaling
#         # get key, value proj
#         if is_cross_attention and past_key_value is not None:
#             # reuse k,v, cross_attentions
#             key_states = past_key_value[0]
#             value_states = past_key_value[1]
#         elif is_cross_attention:
#             # cross_attentions
#             key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
#             value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
#         elif past_key_value is not None:
#             # reuse k, v, self_attention
#             key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
#             value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
#             key_states = torch.cat([past_key_value[0], key_states], dim=2)
#             value_states = torch.cat([past_key_value[1], value_states], dim=2)
#         else:
#             # self_attention
#             key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
#             value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
#
#         if self.is_decoder:
#             # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
#             # Further calls to cross_attention layer can then reuse all cross-attention
#             # key/value_states (first "if" case)
#             # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
#             # all previous decoder key/value_states. Further calls to uni-directional self-attention
#             # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
#             # if encoder bi-directional self-attention `past_key_value` is always `None`
#             past_key_value = (key_states, value_states)
#
#         proj_shape = (bsz * self.num_heads, -1, self.head_dim)
#         query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
#         key_states = key_states.view(*proj_shape)
#         value_states = value_states.view(*proj_shape)
#
#         ##
#         src_len = key_states.size(1)//2
#         ##
#
#         attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
#
#         if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
#                 f" {attn_weights.size()}"
#             )
#
#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, tgt_len, src_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
#
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)
#
#         if layer_head_mask is not None:
#             if layer_head_mask.size() != (self.num_heads,):
#                 raise ValueError(
#                     f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
#                     f" {layer_head_mask.size()}"
#                 )
#             attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
#
#         if output_attentions:
#             # this operation is a bit awkward, but it's required to
#             # make sure that attn_weights keeps its gradient.
#             # In order to do so, attn_weights have to be reshaped
#             # twice and have to be reused in the following
#             attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
#         else:
#             attn_weights_reshaped = None
#
#         attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
#
#         attn_output = torch.bmm(attn_probs, value_states)
#
#         if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )
#
#         attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
#         attn_output = attn_output.transpose(1, 2)
#
#         # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
#         # partitioned aross GPUs when using tensor-parallelism.
#         attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
#
#         attn_output = self.out_proj(attn_output)
#
#         return attn_output, attn_weights_reshaped, past_key_value


class CustomPegasusModel(PegasusPreTrainedModel):
    def __init__(self, config: PegasusConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = PegasusEncoder(config, self.shared)
        self.decoder = PegasusDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        self.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        return (self.encoder.get_position_embeddings(), self.decoder.get_position_embeddings())

    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import PegasusTokenizer, PegasusModel

        >>> tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
        >>> model = PegasusModel.from_pretrained("google/pegasus-large")

        >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
        >>> decoder_inputs = tokenizer("Studies show that", return_tensors="pt")
        >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 4, 1024]
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #
        input_ids = input_ids.repeat_interleave(4, 0)
        attention_mask = attention_mask.repeat_interleave(4, 0).reshape(input_ids.shape)
        #

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # encoder_outputs[0] = encoder_outputs[0].repeat_interleave(4, 0)
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0].repeat_interleave(4, 0),
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        #1 reshape attention_mask, encoder_output[0], decoder_input_ids cho decoder

        # encoder_outputs[0] = encoder_outputs[0]
        decoder_input_ids = decoder_input_ids.reshape(8, -1)
        #1

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "The PEGASUS Model with a language modeling head. Can be used for summarization.", PEGASUS_START_DOCSTRING
)
class CustomPegasusForConditionalGeneration(PegasusPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder.version",
        r"decoder.version",
        r"lm_head.weight",
        r"embed_positions.weight",
    ]

    def __init__(self, config: PegasusConfig):
        super().__init__(config)
        self.model = CustomPegasusModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        self.model.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        return (self.model.encoder.get_position_embeddings(), self.model.decoder.get_position_embeddings())

    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(PEGASUS_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:

            #2 reshape labels để giống với decoder_input_ids
            labels = labels.reshape(8, -1)
            #2

            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            #3 format shape của output của hàm loss để có shape 8x128
            loss_fct = CrossEntropyLoss(size_average=False, reduce=False, reduction='none')
            #3

            #4 tính loss nhưng không flatten mà giữ nguyên shape của labels và logits
            labels[labels==-100.0] = 0.0
            labels = F.one_hot(labels, num_classes=96104)
            lm_logits[labels==0.0] = 0.0
            labels = torch.transpose(labels, 1, 2)
            lm_logits = torch.transpose(lm_logits, 1, 2)
            labels = labels.float()
            masked_lm_loss = loss_fct(lm_logits, labels)
            #4

            #5 reduce mean theo chiều Length, reduce min theo chiều Reference --> MinRefLoss
            masked_lm_loss = masked_lm_loss.reshape(2, 4, -1)
            sum = masked_lm_loss.sum(2, True).squeeze()
            count = torch.count_nonzero(masked_lm_loss, dim=2)
            masked_lm_loss = torch.div(sum, count)
            masked_lm_loss = torch.min(masked_lm_loss, 1, True)
            masked_lm_loss = masked_lm_loss.values.mean()
            #5

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

