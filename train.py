
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

if len(sys.argv)<=1:
    print("Usage: python -u train.py modelname | tee -a logs/log_modelname.txt")
    sys.exit()

########## arguments for training ################################

#segfiles = "data/*.train.seg.aligned"
segfiles = "../WhisperE+Phi2/data/cv.*.train.seg.aligned"
#segfiles = "../WhisperE+Phi2/data_augment/*.train.seg.aligned"

segfiles_dev = segfiles.replace("train","dev")

memory = True
early_stopping = 10

#model_name = "openai/whisper-large-v3"
#model_name = "openai/whisper-medium"
model_name = "openai/whisper-large-v2"

load = None #"saves/model_newwords/checkpoint-99000" # For loading one model but training with new optimizer under other name

eval_steps = 1000 # evaluation every x updates

learning_rate = 2e-4

if not memory:
    model_class = WhisperForConditionalGeneration
    trainer_class = MySeq2SeqTrainer
else:
    model_class = WhisperForConditionalGenerationMemory
    trainer_class = MySeq2SeqTrainerMemory

########## end arguments for training ############################

model_path = sys.argv[1]
output_dir="./saves/"+model_path

checkpoint = glob(output_dir+"/checkpoint*")
resume = len(checkpoint) > 0

dataset = {} #DatasetDict()

if type(segfiles) is not list:
    dataset["train"] = MyDataset(segfiles, memory=memory)
else:
    dataset["train"] = ConcatDataset([MyDataset(s, memory=memory) for s in segfiles])
#dataset["train"][0]
if type(segfiles_dev) is not list:
    dataset["dev"] = MyDataset(segfiles_dev, dev=True, memory=memory)
else:
    dataset["dev"] = ConcatDataset([MyDataset(s, dev=True, memory=memory) for s in segfiles_dev])

print(len(dataset["train"]))
print(len(dataset["dev"]))

tokenizer = WhisperTokenizerFast.from_pretrained(model_name)
#tokenizer.set_prefix_tokens(language="german", task="transcribe")
tokenizer.set_prefix_tokens(task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer)

#data_collator([dataset["train"][i] for i in range(32)])

if not resume and load is None:
    if model_class == WhisperForConditionalGenerationMemory:
        model = model_class.from_pretrained(model_name, torch_dtype="auto", device_map="cuda", init_params=True)
    else:
        model = model_class.from_pretrained(model_name, torch_dtype="auto", device_map="cuda")
else:
    if resume:
        model = model_class.from_pretrained(checkpoint[0], torch_dtype="auto", device_map="cuda")
        print("Resuming training with",checkpoint[0])
    else:
        model = model_class.from_pretrained(load, torch_dtype="auto", device_map="cuda")
        print("Loading checkpoint from",load)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1000000:.0f} M, number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.0f} M")

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=learning_rate,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=500,
    max_steps=200000,
    gradient_checkpointing=False,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=32,
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
    callbacks=[MyCallback]+([EarlyStoppingCallback(early_stopping_patience=early_stopping)] if early_stopping>0 else []),
)

trainer.train(resume_from_checkpoint=resume)

