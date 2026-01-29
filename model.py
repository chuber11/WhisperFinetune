
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

        self.decoder = [decoder]

    def forward(self, memory):
        if memory is None:
            return

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

        #if memory_text_enc is not None:
        memory_text_enc[memory_text_mask.eq(0)] = 0
        encoder_output_memory = 3 * memory_text_enc.sum(1) / lengths # n_mem x d_model

        if "double" in memory and memory["double"]:
            encoder_output_memory_predict = encoder_output_memory[encoder_output_memory.shape[0]//2:] # n_mem x d_model
            encoder_output_memory = encoder_output_memory[:encoder_output_memory.shape[0]//2] # n_mem x d_model
        else:
            encoder_output_memory_predict = encoder_output_memory

        res = [encoder_output_memory, encoder_output_memory_predict]
        
        if not self.training:
            self.last_memory = memory
            self.encoded_memory_cache = res

        return res

@dataclass
class BaseModelOutputMemory(BaseModelOutput):
    memory: Optional[Tuple[torch.FloatTensor, ...]] = None
    add_score: Optional[int] = None
    first_memory_id: Optional[int] = None

@dataclass
class Seq2SeqModelOutputMemory(Seq2SeqModelOutput):
    encoded_memory: Optional[Tuple[torch.FloatTensor, ...]] = None

class WhisperModelMemory(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super(WhisperPreTrainedModel, self).__init__(config)

        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)

        self.encoder_memory = EncoderMemory(self.decoder)

        # Initialize weights and apply final processing
        self.post_init()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.linear = nn.Linear(config.d_model,config.d_model, bias=False)

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
        first_memory_id = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
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
            #print("1) Encoded memory of size", len(memory))

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if "memory" in encoder_outputs and encoder_outputs.memory is not None:
            memory = encoder_outputs.memory
        if "first_memory_id" in encoder_outputs:
            first_memory_id = encoder_outputs.first_memory_id

        #print(f"{decoder_input_ids = }, {first_memory_id = }")

        if memory is not None:
            memory_indices = decoder_input_ids-first_memory_id
            memory_indices.clamp_(min=0)
            decoder_inputs_embeds_mem = memory[1][memory_indices.view(-1)].view(*decoder_input_ids.shape,-1)
            #print(1,decoder_inputs_embeds_mem.std())
            decoder_inputs_embeds_mem = 0.03566*self.linear(decoder_inputs_embeds_mem)
            #print(2,decoder_inputs_embeds_mem.std())

            memory_mask = decoder_input_ids.lt(first_memory_id).to(memory[1].dtype).unsqueeze(-1)
            #print(f"{memory_mask = }")

            decoder_input_ids_c = decoder_input_ids.clone()
            decoder_input_ids_c.clamp_(max=self.decoder.embed_tokens.weight.shape[0]-1)
            decoder_inputs_embeds_nomem = self.decoder.embed_tokens(decoder_input_ids_c)
            #print(3,decoder_inputs_embeds_nomem.std())

            decoder_inputs_embeds = memory_mask * decoder_inputs_embeds_nomem + (1-memory_mask) * decoder_inputs_embeds_mem
        else:
            decoder_input_ids_c = decoder_input_ids.clone()
            decoder_input_ids_c.clamp_(max=self.decoder.embed_tokens.weight.shape[0]-1)
            decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids_c)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
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
            encoded_memory = memory,
        )

@dataclass
class Seq2SeqLMOutputMemory(Seq2SeqLMOutput):
    statistics: Optional[torch.FloatTensor] = None

def get_loss(logits, labels, mask, mean=False): # shapes L x N, L, L
    #if labels.max() >= logits.shape[1] or labels.min() < 0:
    #    print("WARNING: Label indices not in range! Ignoring.")
    #    return 0
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

        self.factor = config.d_model ** -0.25
        self.linear = nn.Linear(config.d_model, config.d_model, bias=False)
        self.linear2 = nn.Linear(config.d_model, config.d_model, bias=False)

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
        first_memory_id = None,
        cache_position = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        #print(f"{decoder_input_ids = }")

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
            first_memory_id=first_memory_id,
        )
        encoded_memory = outputs.encoded_memory

        if memory is not None or encoded_memory is not None:
            mem = self.factor * self.linear(encoded_memory[0]) # n_mem x d_model
            #print("unemb",mem)
            decoder_output_mem = self.factor * self.linear2(outputs[0]) # b x l_tgt x d_model
            lm_logits_mem = torch.matmul(decoder_output_mem, mem.T) # b x l_tgt x n_mem
            #print(1,lm_logits_mem.mean(),lm_logits_mem.std())
            #print("logits_mem",lm_logits_mem)
            #lm_logits_mem = lm_logits_mem - lm_logits_mem.mean(-1,keepdim=True)

            lm_logits_nomem = self.proj_out(outputs[0]) # b x l_tgt x n_vocab
            #print(2,lm_logits_nomem.mean(),lm_logits_nomem.std())
            #print("logits_nomem max",lm_logits_nomem.max(-1)[0])
            #lm_logits_nomem = lm_logits_nomem - lm_logits_nomem.mean(-1,keepdim=True)

            lm_logits = torch.cat([lm_logits_nomem,lm_logits_mem],-1) # b x l_tgt x (n_vocab+n_mem)
        else:
            lm_logits = self.proj_out(outputs[0]) # b x l_tgt x n_vocab

        #print(lm_logits_nomem.shape, lm_logits_mem.shape)
        #print("logits argmax",lm_logits.argmax(-1))

        loss = None
        statistics = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device).reshape(-1)
            loss = loss_fct(lm_logits.view(labels.shape[0], -1), labels)

            statistics = []

            mask = labels.ge(0)
            mask_ = labels.lt(0)
            labels_c = labels.clone()
            labels_c[mask_] = 0
            mask2 = mask & labels.lt(first_memory_id)
            mask3 = mask & labels.ge(first_memory_id)

            logits = lm_logits.view(labels.shape[0], -1).detach()
            add_loss(statistics, logits, labels_c, mask)
            add_loss(statistics, logits, labels_c, mask2)
            add_loss(statistics, logits, labels_c, mask3)

            statistics = torch.stack(statistics)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

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

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor, model_kwargs, model_input_name, generation_config):
        add_score = model_kwargs["memory"].get("add_score", 0) if "memory" in model_kwargs and model_kwargs["memory"] else 0
        if "first_memory_id" in model_kwargs:
            first_memory_id = model_kwargs["first_memory_id"]
        else:
            first_memory_id = model_kwargs["memory"].get("first_memory_id", None) if "memory" in model_kwargs and model_kwargs["memory"] else None

        memory = model_kwargs["memory"] if "memory" in model_kwargs else None
        memory = self.model.encoder_memory(memory)
        #print("2) Encoded memory of size", len(memory[0]))

        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name, generation_config)
        model_kwargs["encoder_outputs"] = BaseModelOutputMemory(*model_kwargs["encoder_outputs"].values(),memory=memory, add_score=add_score, first_memory_id=first_memory_id)
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
            state_dict = torch.load("saves/model_newwords15/checkpoint-184000/encoder_memory.pt")
            model.model.encoder_memory.load_state_dict(state_dict, strict=False)
        else:
            return WhisperForConditionalGenerationMemoryWrapper.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
        return model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

class WhisperForConditionalGenerationConfidence(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        for p in self.parameters():
            p.requires_grad = False

        self.proj_out = nn.Linear(2*config.d_model, 3, bias=False)
        self.post_init()

    def forward(self, *args, first_memory_id=None, **kwargs):
        confidence_labels = kwargs.pop("confidence_labels", None)
        labels = kwargs.pop("labels")

        res = super().forward(*args, **kwargs)
        last_hidden_state = res.last_hidden_state  # b x l_tgt x d_model

        labels_clamp = labels.clamp(min=0)
        label_emb = self.decoder.embed_tokens(labels_clamp)  # b x l_tgt x d_model
        lm_logits = self.proj_out(torch.cat([last_hidden_state,label_emb],-1))  # b x l_tgt x 2

        res.decoder_hidden_states = lm_logits

        loss = None
        statistics = None
        if confidence_labels is not None:
            loss_fct = CrossEntropyLoss()

            confidence_labels = confidence_labels.to(lm_logits.device).reshape(-1)
            lm_logits = lm_logits.view(confidence_labels.shape[0], -1)

            loss = loss_fct(lm_logits, confidence_labels)
            res["loss"] = loss

            statistics = []

            mask = confidence_labels.ge(0)
            confidence_labels = confidence_labels.clamp(min=0)
            add_loss(statistics, lm_logits, confidence_labels, mask) # for acc calc

            statistics = torch.stack(statistics)
            res["statistics"] = statistics

        return res
