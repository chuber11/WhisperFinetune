
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

import subprocess

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
        logs.update(self.compute_mymetrics())
        super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False):
        res = super().compute_loss(model, inputs, return_outputs=True)

        self.statistics[0] += res[0].detach().item()
        self.statistics[1] += 1

        if return_outputs:
            return res
        else:
            return res[0]

    def compute_mymetrics(self):
        logs = {}
        if self.statistics[1] > 0:
            logs["loss_ntp"] = self.statistics[0]/self.statistics[1]
            logs["ppl_ntp"] = math.exp(self.statistics[0]/self.statistics[1])
            self.statistics = [0.0 for _ in range(len(self.statistics))]
        return logs

class MySeq2SeqTrainerMemory(MySeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_statistics=9, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        res = super().compute_loss(model, inputs, return_outputs=True)

        self.statistics[0] += res[1].loss_ntp.detach().item()
        self.statistics[1] += 1
        self.statistics[2] += res[1].loss_memory_total.detach().item()
        self.statistics[3] += res[1].anz_layers.item()
        self.statistics[4] += res[1].loss_memory_mem.detach().item()
        self.statistics[5] += res[1].anz_mem.item()
        self.statistics[6] += res[1].acc_mem.item()
        self.statistics[7] += res[1].acc_nomem.item()
        self.statistics[8] += res[1].anz_total.item()

        if return_outputs:
            return res
        else:
            return res[0]

    def compute_mymetrics(self):
        logs = {}
        if self.statistics[1] > 0:
            logs["loss_ntp"] = self.statistics[0]/self.statistics[1]
            logs["ppl_ntp"] = math.exp(self.statistics[0]/self.statistics[1])
            logs["loss_memory_total"] = self.statistics[2]/self.statistics[3]
            logs["ppl_memory_total"] = math.exp(self.statistics[2]/self.statistics[3])
            logs["acc_memory_nomem"] = self.statistics[7]/(self.statistics[8]-self.statistics[5])
            logs["ppl_memory_mem"] = math.exp(self.statistics[4]/self.statistics[5])
            logs["acc_memory_mem"] = self.statistics[6]/self.statistics[5]
            self.statistics = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        return logs

class MyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs): # Evaluate at the beginning of the training
        if state.global_step == 1:
            pass #control.should_evaluate = True

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
parser.add_argument('--load', type=str, help='For loading one model but training with new     optimizer under other name', default=None)
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

args = parser.parse_args()
print(args)

assert args.eval_steps % args.log_steps == 0

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
processor = WhisperProcessor.from_pretrained(args.model_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer)

#data_collator([dataset["train"][i] for i in range(32)])

possible_factorization = False

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
        model = model_class.from_pretrained(args.load, torch_dtype="auto", device_map="cuda")
        print("Loading checkpoint from",args.load)

        possible_factorization = True

if possible_factorization and args.factorization_rank > 0:
    from peft import get_peft_model, LoraConfig

    peft_config = LoraConfig(inference_mode=False, r=args.factorization_rank, lora_alpha=16, lora_dropout=0.1, bias="all", target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])

    if not args.factorization_only_decoder:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        for p in model.get_encoder().parameters():
            p.requires_grad = False
        model.model.decoder = get_peft_model(model.model.decoder, peft_config)
        model.model.decoder.print_trainable_parameters()

    """from model_factorization import FactorizationWrapper

    if not args.factorization_only_decoder:
        model = FactorizationWrapper(model, args.factorization_rank)
    else:
        for p in model.get_encoder().parameters():
            p.requires_grad = False
        model.model.decoder = FactorizationWrapper(model.model.decoder, args.factorization_rank)"""

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1000000:.0f} M, number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.0f} M")

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps, # increase by 2x for every 2x decrease in batch size
    learning_rate=args.learning_rate,
    lr_scheduler_type="constant_with_warmup",
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
    metric_for_best_model="loss",
    greater_is_better=False,
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

