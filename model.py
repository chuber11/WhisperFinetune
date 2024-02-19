
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from typing import Optional, Tuple, Union

from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperModel
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperDecoder, WhisperDecoderLayer, WhisperPreTrainedModel, WhisperPositionalEmbedding
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration

from dataclasses import dataclass

from transformers.utils import logging

logger = logging.get_logger(__name__)

class AttentionMemory(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(1024, d_model)
        self.v = nn.Linear(1024, d_model)
        self.temperature = np.power(d_model, -0.25)

    def forward(self, hidden_states, memory):
        memory = memory.to(hidden_states.dtype)

        q = self.temperature * self.q(hidden_states) # b x l_tgt x d_model
        k = self.temperature * self.k(memory) # (n_mem+1) x d_model
        v = self.v(memory) # (n_mem+1) x d_model

        attn = torch.einsum("t b d, n d -> t b n", q, k) # b x l_tgt x (n_mem+1)
        probs = F.softmax(attn, -1) # b x l_tgt x (n_mem+1)
        output = torch.matmul(probs, v) # b x l_tgt x d_model

        return attn, output

@dataclass
class BaseModelOutputWithPastAndCrossAttentionsMemory(BaseModelOutputWithPastAndCrossAttentions):
    all_memory_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class WhisperDecoderLayerMemory(WhisperDecoderLayer):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        for p in self.self_attn.parameters():
            p.requires_grad = False
        for p in self.encoder_attn.parameters():
            p.requires_grad = False

        self.memory_attn = AttentionMemory(self.embed_dim)
        self.memory_layer_norm = nn.LayerNorm(self.embed_dim)

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

        outputs = super().forward(hidden_states=hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=layer_head_mask, cross_attn_layer_head_mask=cross_attn_layer_head_mask, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
        hidden_states = outputs[0]

        residual = hidden_states

        hidden_states = self.memory_layer_norm(hidden_states)

        encoder_output_memory, memory_text_enc = memory

        cross_attn_weights, hidden_states = self.memory_attn(hidden_states, encoder_output_memory) # b x l_tgt x (n_mem+1)
        #if cross_attn_weights.argmax(-1).item() > 0:
        #    print(cross_attn_weights,encoder_output_memory.shape)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, *outputs[1:], cross_attn_weights)
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
        del self.model.embed_tokens

        d_model = self.model.layers[0].self_attn.k_proj.in_features
        if d_model != 1280:
            self.linear = nn.Linear(1280, d_model)
        else:
            self.linear = None

        self._no_entry_found = nn.Parameter(torch.randn(1,1024))
        self.decoder = [decoder]

    @property
    def no_entry_found(self):
        nef = self._no_entry_found.to(torch.float32)
        m, s = nef.mean(), nef.std()
        return (nef - m) / s

    def forward(self, memory):
        if memory is None:
            return self.no_entry_found, None

        memory_text_embeds, memory_text_mask = self.decoder[0].embed_tokens(memory["input_ids"]), memory["attention_mask"]
        if self.linear is not None:
            memory_text_embeds = 3 * self.linear(memory_text_embeds)

        lengths = memory_text_mask.eq(1).sum(1).unsqueeze(1) # n_mem x 1

        memory_text_enc = self.model(inputs_embeds=memory_text_embeds, attention_mask=memory_text_mask)[0] # n_mem x l_mem x d_model

        #if memory_text_enc is not None:
        memory_text_enc[memory_text_mask.eq(0)] = 0

        encoder_output_memory_wonef = 3 * memory_text_enc.sum(1) / lengths # n_mem x d_model

        encoder_output_memory = torch.cat([self.no_entry_found, encoder_output_memory_wonef],0) # (n_mem+1) x d_model

        return encoder_output_memory, memory_text_enc

@dataclass
class Seq2SeqModelOutputMemory(Seq2SeqModelOutput):
    all_memory_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class BaseModelOutputMemory(BaseModelOutput):
    memory: Optional[Tuple[torch.FloatTensor, ...]] = None

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

        self.two_decoders = True

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

        if self.two_decoders:
            if past_key_values is None:
                past_key_values_nomem = None
            else:
                past_key_values_nomem = past_key_values[:len(past_key_values)//2]
                past_key_values = past_key_values[len(past_key_values)//2:]

            # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values_nomem,
                inputs_embeds=decoder_inputs_embeds,
                #position_ids=decoder_position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs_memory = self.decoder_memory(
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
            memory=memory,
        )

        if not self.two_decoders:
            decoder_outputs = decoder_outputs_memory
        else:
            mask = torch.stack([c.argmax(-1).gt(0) for c in decoder_outputs_memory['all_memory_cross_attentions']]).any(0) # b x l_tgt
            decoder_outputs["last_hidden_state"][mask] = decoder_outputs_memory["last_hidden_state"][mask]

            decoder_outputs['past_key_values'] = decoder_outputs['past_key_values'] + decoder_outputs_memory['past_key_values']

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
            all_memory_cross_attentions=decoder_outputs_memory.all_memory_cross_attentions,
        )

@dataclass
class Seq2SeqLMOutputMemory(Seq2SeqLMOutput):
    loss_ntp: Optional[torch.FloatTensor] = None
    loss_memory_total: Optional[torch.FloatTensor] = None
    anz_layers: Optional[torch.FloatTensor] = None
    loss_memory_mem: Optional[torch.FloatTensor] = None
    anz_mem: Optional[torch.FloatTensor] = None
    acc_mem: Optional[torch.FloatTensor] = None
    acc_nomem: Optional[torch.FloatTensor] = None
    anz_total: Optional[torch.FloatTensor] = None

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
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        loss_ntp = loss

        loss_memory_total = 0
        anz_layers = torch.zeros(1)

        loss_memory_mem = 0
        anz_mem = torch.zeros(1)

        acc_mem = torch.zeros(1)
        acc_nomem = torch.zeros(1)
        anz_total = torch.zeros(1)

        if memory_labels is not None:
            loss_fct_mem = CrossEntropyLoss(ignore_index=0, reduction="sum")
            memory_labels = memory_labels.to(lm_logits.device).reshape(-1)

            for cross_attn_weights in outputs.all_memory_cross_attentions:
                cross_attn_weights = cross_attn_weights.view(-1, cross_attn_weights.shape[-1])
                loss_memory_total_ = loss_fct(cross_attn_weights, memory_labels)
                loss_memory_total = loss_memory_total + loss_memory_total_
                anz_layers += 1

            loss = loss + loss_memory_total

            loss_memory_mem += loss_fct_mem(cross_attn_weights, memory_labels)

            pred_mem_entries = cross_attn_weights.argmax(-1)
            mask = pred_mem_entries.eq(memory_labels)

            mask_mem = memory_labels.gt(0)
            mask_nomem = memory_labels.eq(0)

            acc_mem += mask[mask_mem].sum().to(acc_mem.device)
            acc_nomem += mask[mask_nomem].sum().to(acc_nomem.device)

            anz_mem += mask_mem.sum().to(anz_mem.device)
            anz_total += pred_mem_entries.shape[0]

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
            loss_ntp = loss_ntp,
            loss_memory_total = loss_memory_total,
            anz_layers = anz_layers,
            loss_memory_mem = loss_memory_mem,
            anz_mem = anz_mem,
            acc_mem = acc_mem,
            acc_nomem = acc_nomem,
            anz_total = anz_total,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None):
        if not "encoder_outputs_memory" in model_kwargs:
            model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name)
        else:
            model_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs_memory"]
            del model_kwargs["encoder_outputs_memory"]
        memory = model_kwargs["memory"] if "memory" in model_kwargs else None
        memory = self.model.encoder_memory(memory)
        print("2) Encoded memory of size", len(memory[0])-1)
        model_kwargs["encoder_outputs"] = BaseModelOutputMemory(*model_kwargs["encoder_outputs"].values(),memory=memory)
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

