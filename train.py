
from data import MyDataset, ConcatDataset, DataCollatorSpeechSeq2SeqWithPadding, compute_metrics
from model import WhisperForConditionalGenerationMemory

from transformers import WhisperTokenizerFast
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback

import torch
import math

import sys
from glob import glob
import os
import json

import subprocess

from peft import get_peft_model, LoraConfig, PeftModel

def get_git_commit_hash():
    try:
        # Run the git command to get the commit hash
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return git_hash.decode('utf-8')  # Convert bytes to string
    except subprocess.CalledProcessError:
        return "Not a git repository or no commit found"

# Call the function to get the git commit hash
commit_hash = get_git_commit_hash()
print("############ BEGIN GIT COMMIT HASH ############")
print("Git commit hash:", commit_hash)
print("############ END GIT COMMIT HASH ############")

def print_git_diff():
    try:
        # Execute git diff command
        diff_output = subprocess.check_output(["git", "diff"])

        # Decode the byte output to string and print
        print(diff_output.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        # Handle any errors if the git command fails
        print("Error executing git diff:", e)

# Call the function
print("############ BEGIN GIT DIFF ############")
print_git_diff()
print("############ END GIT DIFF ############")

class MySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, num_statistics=2, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.statistics = [0.0 for _ in range(num_statistics)]
        self.label_names = ["labels"]

    def log(self, logs):
        dev = "eval_loss" in logs
        logs.update(self.compute_mymetrics(dev))
        super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False):
        res = super().compute_loss(model, inputs, return_outputs=True)

        self.statistics[0] += res[0].detach().item()
        self.statistics[1] += 1

        if return_outputs:
            return res
        else:
            return res[0]

    def compute_mymetrics(self, dev):
        logs = {}
        if self.statistics[1] > 0:
            prefix = "" if not dev else "eval_"
            logs[prefix+"loss_ntp"] = self.statistics[0]/self.statistics[1]
            logs[prefix+"ppl_ntp"] = math.exp(self.statistics[0]/self.statistics[1])
            self.statistics = [0.0 for _ in range(len(self.statistics))]
        return logs

    def _save(self, output_dir, **kwargs):
        super()._save(output_dir, **kwargs)

        if hasattr(self.model, "save_embedding_separate") and self.model.save_embedding_separate:
            embedding = {k:v for k,v in self.model.base_model.state_dict().items() if "embed_tokens" in k or "proj_out" in k}
            torch.save(embedding, os.path.join(output_dir, "embedding.pt"))

class MySeq2SeqTrainerMemory(MySeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.statistics = None
        
    def compute_loss(self, model, inputs, return_outputs=False):
        res = super(Seq2SeqTrainer, self).compute_loss(model, inputs, return_outputs=True)

        if self.statistics is None:
            self.statistics = res[1].statistics.cpu().tolist()
        else:
            s_new = res[1].statistics.cpu().tolist()
            for i in range(len(self.statistics)):
                self.statistics[i] += s_new[i]

        if return_outputs:
            return res
        else:
            return res[0]

    def compute_mymetrics(self, dev):
        logs = {}
        index = 0
        prefix = "" if not dev else "eval_"
        for l in ["_ntp","_mem"]:
            for m in ["_all","_nomem","_mem"]:
                if self.statistics[index+2] > 0:
                    logs[prefix+"loss"+l+m] = self.statistics[index]/self.statistics[index+2]
                    logs[prefix+"ppl"+l+m] = math.exp(self.statistics[index]/self.statistics[index+2])
                    logs[prefix+"acc"+l+m] = self.statistics[index+1]/self.statistics[index+2]
                index += 3
        logs = {k:v for k,v in sorted(list(logs.items()))}
        for i in range(len(self.statistics)):
            self.statistics[i] = 0
        return logs

class MyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs): # Evaluate at the beginning of the training
        if state.global_step == 1:
            pass #control.should_evaluate = True

def add_lora(model, factorization_rank, factorization_only_decoder):
    if not factorization_only_decoder:
        peft_config = LoraConfig(inference_mode=False, r=factorization_rank, lora_alpha=16, lora_dropout=0.1, bias="all", target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    else:
        for p in model.get_encoder().parameters():
            p.requires_grad = False
        names = [".".join(n.split(".")[:-1]) for n,p in model.named_parameters() if "decoder" in n and not "layer_norm" in n and not "embed" in n]
        peft_config = LoraConfig(inference_mode=False, r=factorization_rank, lora_alpha=16, lora_dropout=0.1, bias="all", target_modules=names)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--segfiles', type=str, nargs="+", help='Data used for training', default=["../WhisperE+Phi2/data/cv.*.train.seg.aligned"])
parser.add_argument('--dataset_factors', type=int, nargs="+", help='Upscaling factors for datasets', default=None)
parser.add_argument('--segfiles_dev', type=str, nargs="+", help='Data used for evaluation', default=["../WhisperE+Phi2/data/cv.*.dev.seg.aligned"])
parser.add_argument('--dataset_factors_dev', type=int, nargs="+", help='Upscaling factors for evaluation datasets', default=None)
parser.add_argument('--use_memory', action="store_true", help='Generate memory in dataset')
parser.add_argument('--use_early_stopping', type=int, help='Use early stopping', default=10)
parser.add_argument('--model_name', type=str, help='Model architecture to train', default="openai/whisper-large-v2")
parser.add_argument('--model_path', type=str, help='Path to store the trained model', default="./saves/model_test")
parser.add_argument('--load', type=str, help='For loading one model but training with new optimizer under other name', default=None)
parser.add_argument('--log_steps', type=int, help='Print log every x steps', default=10)
parser.add_argument('--eval_steps', type=int, help='Run cross validation every x steps', default=1000)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--gradient_checkpointing', action="store_true", help='Use gradient_checkpointing')
parser.add_argument('--factorization_rank', type=int, help='Factorization rank', default=0)
parser.add_argument('--factorization_only_decoder', action="store_true", help='Use factorization only on decoder and freeze encoder')
parser.add_argument('--factorization_only_save_factorization_parameters', action="store_true", help='Only save the factorization parameters')
parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps', default=4)
parser.add_argument('--warmup_steps', type=int, help='Warmup steps', default=500)

parser.add_argument('--metric_for_best_model', type=str, help='Which metric to use to determine the best model', default="loss")
parser.add_argument('--greater_is_better', action="store_true", help='If higher metric is better')
parser.add_argument('--only_train_embedding', action="store_true", help='Freeze all weights except the projection layer / embedding layer')
parser.add_argument('--train_embedding', action="store_true", help='Train embedding / proj layer weights')
parser.add_argument('--freeze_encoder', action="store_true", help='Freeze the encoder parameters')

args = parser.parse_args()
print(args)

assert args.eval_steps % args.log_steps == 0

if args.load == "None":
    args.load = None

if not args.use_memory:
    model_class = WhisperForConditionalGeneration
    trainer_class = MySeq2SeqTrainer
else:
    model_class = WhisperForConditionalGenerationMemory
    trainer_class = MySeq2SeqTrainerMemory

output_dir=args.model_path

checkpoint = glob(output_dir+"/checkpoint*")
resume = len(checkpoint) > 0

dataset = {} #DatasetDict()

if len(args.segfiles) == 1 and args.dataset_factors is None:
    dataset["train"] = MyDataset(args.segfiles[0], memory=args.use_memory)
else:
    dataset["train"] = ConcatDataset([MyDataset(s, memory=args.use_memory) for s in args.segfiles], factors=args.dataset_factors)
#dataset["train"][0]
if len(args.segfiles_dev) == 1 and args.dataset_factors_dev is None:
    dataset["dev"] = MyDataset(args.segfiles_dev[0], dev=True, memory=args.use_memory)
else:
    dataset["dev"] = ConcatDataset([MyDataset(s, dev=True, memory=args.use_memory) for s in args.segfiles_dev], factors=args.dataset_factors_dev)

tokenizer = WhisperTokenizerFast.from_pretrained(args.model_name)
#tokenizer.set_prefix_tokens(language="german", task="transcribe")
tokenizer.set_prefix_tokens(task="transcribe")
tokenizer.pad_token = tokenizer.eos_token
processor = WhisperProcessor.from_pretrained(args.model_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer)

#data_collator([dataset["train"][i] for i in range(32)])

possible_factorization = False

load_adapter = None
if not resume and args.load is None:
    if model_class == WhisperForConditionalGenerationMemory:
        model = model_class.from_pretrained(args.model_name, torch_dtype="auto", device_map="cuda", init_params=True)
    else:
        model = model_class.from_pretrained(args.model_name, torch_dtype="auto", device_map="cuda")

    possible_factorization = True
else:
    if resume:
        model = model_class.from_pretrained(checkpoint[0], torch_dtype="auto", device_map="cuda")
        print("Resuming training with",checkpoint[0])
    else:
        files = glob(args.load+"/*/config.json")
        if len(files) == 0:
            files = glob(args.load+"/*/adapter_config.json")
            if len(files) != 1:
                print(files)
                raise RuntimeError
            load_adapter = args.load
            args.load = json.load(open(files[0]))["base_model_name_or_path"]
        elif len(files) != 1:
            breakpoint()
        checkpoint = "/".join(files[0].split("/")[:-1])
        print("Loading checkpoint from",checkpoint)
        model = model_class.from_pretrained(checkpoint, torch_dtype="auto", device_map="cuda")

        #for p in model.proj_out.parameters():
        #    p.requires_grad = True

        possible_factorization = True

if load_adapter is not None:
    files = glob(load_adapter+"/*")
    if len(files) != 1:
        print(files)
        raise RuntimeError
    model = PeftModel.from_pretrained(model, files[0])
    model = model.merge_and_unload()
    print("Loaded adapter from",files[0])

factorization = possible_factorization and args.factorization_rank > 0

if factorization:
    model = add_lora(model, args.factorization_rank, args.factorization_only_decoder)

if args.only_train_embedding:
    for p in model.parameters():
        p.requires_grad = False
if args.only_train_embedding or args.train_embedding:
    for p in model.proj_out.parameters():
        p.requires_grad = True

    if factorization:
        model.save_embedding_separate = True

if args.freeze_encoder:
    for p in model.get_encoder().parameters():
        p.requires_grad = False

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1000000:.0f} M, number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.0f} M")

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps, # increase by 2x for every 2x decrease in batch size
    learning_rate=args.learning_rate,
    lr_scheduler_type="linear", #"constant_with_warmup",
    warmup_steps=args.warmup_steps,
    max_steps=200000,
    gradient_checkpointing=args.gradient_checkpointing,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=args.batch_size,
    predict_with_generate=False,#True,
    save_steps=args.eval_steps,
    eval_steps=args.eval_steps,
    logging_steps=args.log_steps,
    save_total_limit=1,
    #report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model=args.metric_for_best_model,
    greater_is_better=args.greater_is_better,
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_num_workers=8,
)

trainer = trainer_class(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #tokenizer=processor.feature_extractor,
    #tokenizer=processor.tokenizer,
    callbacks=[MyCallback]+([EarlyStoppingCallback(early_stopping_patience=args.use_early_stopping)] if args.use_early_stopping>0 else []),
)

trainer.train(resume_from_checkpoint=resume)

