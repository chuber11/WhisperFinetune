
from data import MyDataset, DataCollatorCTCWithPadding, compute_metrics

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2ForCTC

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import TrainerCallback

import math

import sys
from glob import glob

class MySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.statistics = [0.0,0.0]

    #def evaluate(self, *args, **kwargs):
    #    kwargs["max_length"] = max_length
    #    kwargs["no_repeat_ngram_size"] = 6
    #    kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    #    return super().evaluate(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        res = super().compute_loss(model, inputs, return_outputs=True)

        if model.training:
            self.statistics[0] += res[0].detach()
            self.statistics[1] += 1

        if return_outputs:
            return res
        else:
            return res[0]

    def log(self, logs):
        if not "eval_ppl" in logs:
            if self.statistics[1] > 0:
                logs["ppl"] = math.exp(self.statistics[0].item()/self.statistics[1])
                self.statistics = [0.0,0.0]
        super().log(logs)

class MyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs): # Evaluate at the beginning of the training
        if state.global_step == 1:
            pass #control.should_evaluate = True

augment = False

dataset = {} #DatasetDict()
dataset["train"] = MyDataset(augment=augment)
dataset["test"] = MyDataset(dev=True, augment=augment)

print(len(dataset["train"]))
print(len(dataset["test"]))

model_name = "facebook/wav2vec2-large-xlsr-53"

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

if len(sys.argv)<=1:
    print("You have to specify a model name")
    sys.exit()
model_path = sys.argv[1]
output_dir="./saves/"+model_path

checkpoint = glob(output_dir+"/checkpoint*")
resume = len(checkpoint) > 0

load = None

from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import CausalLMOutput
import torch.nn as nn

class MyWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


if not resume and load is None:
    model = MyWav2Vec2ForCTC.from_pretrained(model_name, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id, vocab_size=tokenizer.vocab_size+3)
else:
    if resume:
        model = MyWav2Vec2ForCTC.from_pretrained(checkpoint[0], ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id, vocab_size=tokenizer.vocab_size+3)
        print("Resuming training with",checkpoint[0])
    else:
        model = MyWav2Vec2ForCTC.from_pretrained(load, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id, vocab_size=tokenizer.vocab_size+3)
        print("Loading checkpoint from",load)

model.freeze_feature_extractor()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1000000:.0f} M, number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.0f} M")

eval_steps = 45

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-4, #8e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=500,
    max_steps=100000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=False,#True,
    save_steps=eval_steps,
    eval_steps=eval_steps,
    logging_steps=10,
    save_total_limit=1,
    #report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_num_workers=4,
)

print("TODO: more dataloader workers?")

trainer = MySeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    #tokenizer=processor.tokenizer,
    callbacks=[MyCallback],
)

trainer.train(resume_from_checkpoint=resume)

