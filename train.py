
from data import MyDataset, DataCollatorSpeechSeq2SeqWithPadding, compute_metrics

from transformers import WhisperTokenizerFast
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import TrainerCallback

import torch
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

model_name = "openai/whisper-large-v3"
#model_name = "openai/whisper-medium"

tokenizer = WhisperTokenizerFast.from_pretrained(model_name)
tokenizer.set_prefix_tokens(language="german", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer)

if len(sys.argv)<=1:
    print("You have to specify a model name")
    sys.exit()
model_path = sys.argv[1]
output_dir="./saves/"+model_path

checkpoint = glob(output_dir+"/checkpoint*")
resume = len(checkpoint) > 0

load = None

if not resume and load is None:
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cuda")
else:
    if resume:
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint[0], torch_dtype="auto", device_map="cuda")
        print("Resuming training with",checkpoint[0])
    else:
        model = WhisperForConditionalGeneration.from_pretrained(load, torch_dtype="auto", device_map="cuda")
        print("Loading checkpoint from",load)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1000000:.0f} M, number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.0f} M")

eval_steps = 45

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5, #8e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=500,
    max_steps=1000,
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

trainer = MySeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #tokenizer=processor.feature_extractor,
    #tokenizer=processor.tokenizer,
    callbacks=[MyCallback],
)

trainer.train(resume_from_checkpoint=resume)

