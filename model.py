
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from typing import Optional, Tuple, Union

from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperModel
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperDecoder, WhisperDecoderLayer, WhisperPreTrainedModel, WhisperPositionalEmbedding, WHISPER_ATTENTION_CLASSES
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration

import inspect
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper

from dataclasses import dataclass

from transformers.utils import logging

logger = logging.get_logger(__name__)

class AttentionMemory(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.temperature = np.power(d_model, -0.25)

    def forward(self, hidden_states, memory):
        memory = memory.to(hidden_states.dtype)

        q = self.temperature * self.q(hidden_states) # b x l_tgt x d_model
        k = self.temperature * self.k(memory) # (n_mem+1) x d_model

        attn = torch.einsum("t b d, n d -> t b n", q, k) # b x l_tgt x (n_mem+1)
        return attn

@dataclass
class BaseModelOutputWithPastAndCrossAttentionsMemory(BaseModelOutputWithPastAndCrossAttentions):
    all_memory_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class WhisperDecoderLayerMemory(WhisperDecoderLayer):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        for module in [self.self_attn, self.encoder_attn, self.fc1, self.fc2]:
        #for module in [self.self_attn, self.encoder_attn]:
            for p in module.parameters():
                p.requires_grad = False

        self.memory_attn = AttentionMemory(self.embed_dim)
        self.memory_layer_norm = nn.LayerNorm(self.embed_dim)

        self.entry_norm = False
        if self.entry_norm:
            self.memory_entry_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.memory_entry_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )

    def calc_memory_entry_attn(self, dec_output, mem_attn_out, memory_text_enc, memory_text_mask):
        b, l_tar, _ = mem_attn_out.shape

        if memory_text_enc is None:
            return None,None

        mem_attn_out = mem_attn_out.argmax(-1).view(-1) - 1 # b*l_tar

        # filter -1´s
        mask = mem_attn_out.ne(-1)
        if mask.any():
            indices = torch.arange(mask.shape[0], device=mask.device)[mask]
            mem_attn_out = mem_attn_out[mask]

            dec_output = dec_output.view(b*l_tar, -1) # b*l_tar x d_model
            hidden_states = dec_output[indices].unsqueeze(1) # mask.sum() x 1 x d_model

            key_value_states = memory_text_enc[mem_attn_out] # mask.sum() x l_mem x d_model
            attention_mask = memory_text_mask[mem_attn_out].unsqueeze(1).unsqueeze(1) # mask.sum() x 1 x 1 x l_mem

            memory_entry_attn, _, memory_attn_present_key_value = self.memory_entry_attn(hidden_states=hidden_states, key_value_states=key_value_states, attention_mask=attention_mask) # mask.sum() x 1 x d_model
            if self.entry_norm:
                memory_entry_attn = self.memory_entry_attn_layer_norm(memory_entry_attn)

            output = torch.zeros_like(dec_output, dtype=memory_entry_attn.dtype) # b*l_tar x d_model
            output[indices] = memory_entry_attn[:,0]

            return output.view(b, l_tar, -1), memory_attn_present_key_value # b x l_tar x d_model
        else:
            return None, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        memory = None):

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Memory-Attention
        residual = hidden_states
        hidden_states = self.memory_layer_norm(hidden_states)

        encoder_output_memory, memory_text_enc, memory_text_embeds, memory_text_mask, _ = memory

        memory_attn_weights = self.memory_attn(hidden_states, encoder_output_memory) # b x l_tgt x (n_mem+1)
        #if memory_attn_weights.argmax(-1).gt(0).any():
        #    print(memory_attn_weights.argmax(-1))
        #    print(F.softmax(memory_attn_weights,-1)[:,:,-1])
        
        hidden_states, memory_attn_present_key_value = self.calc_memory_entry_attn(dec_output=hidden_states,
                                                    mem_attn_out=memory_attn_weights,
                                                    memory_text_enc=memory_text_embeds,
                                                    memory_text_mask=memory_text_mask)

        if hidden_states is not None:
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
        else:
            hidden_states = residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        outputs += (memory_attn_weights,)
        return outputs

class WhisperDecoderMemory(WhisperDecoder):
    def __init__(self, config: WhisperConfig, decoder):
        super(WhisperPreTrainedModel, self).__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = [decoder.embed_tokens] #nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([WhisperDecoderLayerMemory(config) for _ in range(config.decoder_layers)])

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        memory=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens[0](input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and head_mask is None and not output_attentions:
            # output_attentions=True & head_mask can not be supported when using SDPA.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids, past_key_values_length=past_key_values_length #, position_ids=position_ids
            )
        else:
            positions = self.embed_positions(
                inputs_embeds, past_key_values_length=past_key_values_length #, position_ids=position_ids
            )

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        all_memory_cross_attentions = ()
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                    output_attentions,
                    use_cache,
                    memory,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    memory=memory,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

            all_memory_cross_attentions += (layer_outputs[-1],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions, all_memory_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentionsMemory(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            all_memory_cross_attentions=all_memory_cross_attentions,
        )

class EncoderMemory(nn.Module):
    def __init__(self, decoder):
        super().__init__()

        model_name = "facebook/mbart-large-50"
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).model.encoder
        d_model = self.model.embed_tokens.embedding_dim
        del self.model.embed_tokens

        if d_model != decoder.embed_tokens.embedding_dim:
            self.linear = nn.Linear(decoder.embed_tokens.embedding_dim, d_model)
            self.linear2 = nn.Linear(d_model, decoder.embed_tokens.embedding_dim)
        else:
            self.linear = None
            self.linear2 = None

        self._no_entry_found = nn.Parameter(torch.randn(1,decoder.embed_tokens.embedding_dim))
        self.decoder = [decoder]

    @property
    def no_entry_found(self):
        nef = self._no_entry_found.to(torch.float32)
        m, s = nef.mean(), nef.std()
        return (nef - m) / s

    def forward(self, memory):
        if memory is None:
            return self.no_entry_found, None, None, None

        if not self.training and \
           hasattr(self, "last_memory") and \
           self.last_memory["input_ids"].shape == memory["input_ids"].shape and \
           self.last_memory["input_ids"].eq(memory["input_ids"]).all():
            #print("Using encoded memory cache")
            return self.encoded_memory_cache

        memory_text_embeds, memory_text_mask = self.decoder[0].embed_tokens(memory["input_ids"]), memory["attention_mask"]
        if self.linear is not None:
            memory_text_embeds_ = 3 * self.linear(memory_text_embeds)

        lengths = memory_text_mask.eq(1).sum(1).unsqueeze(1) # n_mem x 1

        memory_text_enc = self.model(inputs_embeds=memory_text_embeds_, attention_mask=memory_text_mask)[0] # n_mem x l_mem x d_model

        if self.linear2 is not None:
            memory_text_enc = self.linear2(memory_text_enc)

        memory_text_mask = memory_text_mask.eq(0)

        #if memory_text_enc is not None:
        memory_text_enc[memory_text_mask] = 0

        memory_text_mask = memory_text_mask.to(memory_text_enc.dtype)
        memory_text_mask[memory_text_mask.eq(1)] = -float("Inf")

        encoder_output_memory_wonef = 3 * memory_text_enc.sum(1) / lengths # n_mem x d_model

        if memory["double"]:
            encoder_output_memory_wonef = encoder_output_memory_wonef[:encoder_output_memory_wonef.shape[0]//2]
            memory_text_enc = memory_text_enc[:memory_text_enc.shape[0]//2]
            memory_text_embeds = memory_text_embeds[:memory_text_embeds.shape[0]//2]
            memory_text_mask = memory_text_mask[:memory_text_mask.shape[0]//2]
            memory_text_ids = memory["input_ids"][memory["input_ids"].shape[0]//2:]
        else:
            memory_text_ids = None

        encoder_output_memory = torch.cat([self.no_entry_found, encoder_output_memory_wonef],0) # (n_mem+1) x d_model

        res = [encoder_output_memory, memory_text_enc, memory_text_embeds, memory_text_mask, memory_text_ids]
        
        if not self.training:
            self.last_memory = memory
            self.encoded_memory_cache = res

        return res

@dataclass
class Seq2SeqModelOutputMemory(Seq2SeqModelOutput):
    all_memory_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class BaseModelOutputMemory(BaseModelOutput):
    memory: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_outputs_memory: Optional[Tuple[torch.FloatTensor, ...]] = None
    add_score: Optional[int] = None

class WhisperModelMemory(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super(WhisperPreTrainedModel, self).__init__(config)

        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)

        self.encoder_memory = EncoderMemory(self.decoder)
        self.decoder_memory = WhisperDecoderMemory(config, self.decoder)

        # Initialize weights and apply final processing
        self.post_init()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_outputs_memory: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        memory = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutputMemory]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            encoder_outputs_memory = encoder_outputs

            memory = self.encoder_memory(memory)
            #print("1) Encoded memory of size", len(memory[0])-1)

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if "memory" in encoder_outputs and encoder_outputs.memory is not None:
            memory = encoder_outputs.memory

        if encoder_outputs_memory is None:
            if encoder_outputs.encoder_outputs_memory is None:
                encoder_outputs_memory = encoder_outputs
            else:
                encoder_outputs_memory = encoder_outputs.encoder_outputs_memory

        if past_key_values is not None and memory[1] is not None:
            past_key_values_mem = past_key_values[len(past_key_values)//2:]
            past_key_values = past_key_values[:len(past_key_values)//2]
        else:
            past_key_values_mem = None

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            #position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        all_memory_cross_attentions = None

        if memory[1] is not None: # non empty memory
            # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
            decoder_outputs_memory = self.decoder_memory(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs_memory[0],
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values_mem,
                inputs_embeds=decoder_inputs_embeds,
                #position_ids=decoder_position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                memory=memory,
            )

            #mask = torch.stack([c.argmax(-1).gt(0) for c in decoder_outputs_memory['all_memory_cross_attentions']]).any(0) # b x l_tgt
            mask = decoder_outputs_memory['all_memory_cross_attentions'][-1].argmax(-1).gt(0) # b x l_tgt
            decoder_outputs["last_hidden_state"][mask] = decoder_outputs_memory["last_hidden_state"][mask]

            decoder_outputs['past_key_values'] = decoder_outputs['past_key_values'] + decoder_outputs_memory['past_key_values']

            all_memory_cross_attentions = decoder_outputs_memory.all_memory_cross_attentions

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutputMemory(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            all_memory_cross_attentions=all_memory_cross_attentions,
        )

@dataclass
class Seq2SeqLMOutputMemory(Seq2SeqLMOutput):
    statistics: Optional[torch.FloatTensor] = None

def get_loss(logits, labels, mask, mean=False): # shapes L x N, L, L
    #if labels.max() >= logits.shape[1] or labels.min() < 0:
        #print("WARNING: Label indices not in range! Ignoring.")
        #return 0
    if not mean:
        return -F.log_softmax(logits, -1).gather(1, labels.unsqueeze(-1))[:,0][mask].sum()
    else:
        return -F.log_softmax(logits, -1).gather(1, labels.unsqueeze(-1))[:,0][mask].mean()

def add_loss(statistics, logits, labels, mask):
    loss = get_loss(logits, labels, mask)
    acc = logits.argmax(-1).eq(labels)[mask].sum()
    anz = mask.sum()
    statistics.append(loss)
    statistics.append(acc)
    statistics.append(anz)

class WhisperForConditionalGenerationMemoryWrapper(WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModelMemory(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        for p in self.proj_out.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        memory = None,
        memory_labels = None,
        encoder_outputs_memory = None, # always only dummy
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutputMemory]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        #print(decoder_input_ids)

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            memory=memory,
        )
        lm_logits = self.proj_out(outputs[0]) # b x l_tgt x n_vocab

        #print(f"{decoder_input_ids = }")

        if encoder_outputs and "memory" in encoder_outputs: # Replace word by memory entry
            memory_ = encoder_outputs["memory"]
            memory_text_ids = encoder_outputs["memory"][4]
            if type(encoder_outputs["memory"]) is list and len(encoder_outputs["memory"]) >= 5 and memory_text_ids is not None:
                if past_key_values is None: # create indices
                    indices = torch.full((lm_logits.shape[0],2), -1,
                                         device=lm_logits.device)
                else: # read last indices
                    indices = past_key_values[-1][0]

                mask = outputs['all_memory_cross_attentions'][-1][:,-1].argmax(-1) # b
                mask2 = mask.gt(0)
                if mask2.any():
                    mask_ = mask2 & indices[:,0].eq(-1)

                    #print(f"{mask  = }")
                    #print(f"{mask_ = }")
                    #print(f"before {indices = }")

                    indices[:,0][mask_] = mask[mask_]-1
                    indices[:,1][mask_] = 0

                mask = indices[:,0].ne(-1)
                if mask.any():
                    # use indices
                    index1 = indices[:,0][mask]
                    tokens = memory_text_ids[index1]
                    index2 = indices[:,1][mask]
                    tokens2 = tokens.gather(1,index2.unsqueeze(-1)) # sum(mask) x 1
                    lm_logits_ = lm_logits[mask]
                    boost_all = False
                    eos_token = 50257
                    if not boost_all:
                        lm_logits_[:,0].scatter_add_(1,tokens2,torch.full_like(tokens2, encoder_outputs.add_score if encoder_outputs.add_score else 0, dtype=lm_logits.dtype))
                    else:
                        add = torch.full_like(tokens, encoder_outputs.add_score if encoder_outputs.add_score else 0, dtype=lm_logits.dtype)
                        add[tokens.eq(eos_token)] = 0
                        lm_logits_[:,0].scatter_add_(1,tokens,add)
                    lm_logits[mask] = lm_logits_

                    tokens2 = tokens.gather(1,(index2+1).unsqueeze(-1)) # sum(mask) x 1
                    mask2 = tokens2.eq(eos_token)
                    if mask2.any():
                        indices_ = indices[:,0][mask]
                        indices_[mask2[:,0]] = -1
                        indices[:,0][mask] = indices_

                indices[:,1][indices[:,1].gt(-1)] += 1

                if mask.any():
                    pass #print(f"after  {indices = }")

                outputs['past_key_values'] = tuple([*outputs['past_key_values'], (indices,)])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device).reshape(-1)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        statistics = None
        if memory_labels is not None:
            factor_mem = 1e-2 #1e-1

            memory_labels = memory_labels.to(lm_logits.device).reshape(-1)
            mask = memory_labels.ge(0)
            mask_ = memory_labels.lt(0)

            for cross_attn_weights in outputs.all_memory_cross_attentions:
                cross_attn_weights = cross_attn_weights.view(-1, cross_attn_weights.shape[-1])
                loss = loss + factor_mem*loss_fct(cross_attn_weights, memory_labels)

            labels_c = labels.clone()
            labels_c[mask_] = 0
            memory_labels_c = memory_labels.clone()
            memory_labels_c[mask_] = 0
            mask2 = mask & memory_labels_c.eq(0)
            mask3 = mask & memory_labels_c.gt(0)

            statistics = []

            logits = lm_logits.view(-1, self.config.vocab_size).detach()
            add_loss(statistics, logits, labels_c, mask)
            add_loss(statistics, logits, labels_c, mask2)
            add_loss(statistics, logits, labels_c, mask3)

            logits2 = cross_attn_weights.detach()
            add_loss(statistics, logits2, memory_labels_c, mask)
            add_loss(statistics, logits2, memory_labels_c, mask2)
            add_loss(statistics, logits2, memory_labels_c, mask3)

            statistics = torch.stack(statistics)

        return Seq2SeqLMOutputMemory(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            statistics=statistics,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None):
        if not "encoder_outputs_memory" in model_kwargs:
            model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name)
        else:
            model_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs_memory"]
            del model_kwargs["encoder_outputs_memory"]
        encoder_outputs_memory = None
        if hasattr(self.model.encoder.layers[0].self_attn.k_proj,"lora_A"): # LORA -> have to run encoder without lora weights
            #print("Running encoder twice...")

            encoder = self.get_encoder()
            irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not any(argument.startswith(p) for p in irrelevant_prefix)
            }
            encoder_signature = set(inspect.signature(encoder.forward).parameters)
            encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
            if not encoder_accepts_wildcard:
                encoder_kwargs = {
                    argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
                }
            model_input_name = model_input_name if model_input_name is not None else self.main_input_name
            encoder_kwargs["return_dict"] = True
            encoder_kwargs[model_input_name] = inputs_tensor
            for name, module in encoder.named_modules():
                if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                    module.enable_adapters(enabled=False)
            encoder_outputs_memory = encoder(**encoder_kwargs)
            for name, module in encoder.named_modules():
                if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                    module.enable_adapters(enabled=True)
        add_score = model_kwargs["memory"].get("add_score", 0) if "memory" in model_kwargs else 0
        memory = model_kwargs["memory"] if "memory" in model_kwargs else None
        memory = self.model.encoder_memory(memory)
        #print("2) Encoded memory of size", len(memory[0])-1)
        model_kwargs["encoder_outputs"] = BaseModelOutputMemory(*model_kwargs["encoder_outputs"].values(),memory=memory, encoder_outputs_memory=encoder_outputs_memory, add_score=add_score)
        #print(model_kwargs["encoder_outputs"])
        return model_kwargs
        
class WhisperForConditionalGenerationMemory(nn.Module):
    @classmethod
    def from_pretrained(cls, model_name, torch_dtype="auto", device_map="cuda", init_params=False):
        if init_params:
            model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
            state_dict = model.state_dict()
            config = model.config
            model = WhisperForConditionalGenerationMemoryWrapper(config)
            model.load_state_dict(state_dict, strict=False)
            model.model.decoder_memory.load_state_dict(model.model.decoder.state_dict(), strict=False)
        else:
            return WhisperForConditionalGenerationMemoryWrapper.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
        return model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

