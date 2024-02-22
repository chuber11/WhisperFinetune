
import torch

from data import MyDataset, DataCollatorSpeechSeq2SeqWithPadding, ConcatDataset

from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast
from model import WhisperForConditionalGenerationMemory

from torch.cuda.amp import autocast

from tqdm import tqdm

import sys
from glob import glob
import os

import argparse

from peft import PeftModel

parser = argparse.ArgumentParser()

parser.add_argument('--segfiles', type=str, nargs="+", help='Data used for training', default=["../WhisperE+Phi2/data_test/cv.*.test.seg.aligned"])
parser.add_argument('--decode_only_first_part', action="store_true")
parser.add_argument('--model_path', type=str, help='Path to store the trained model', default="./saves/model_newwords3/checkpoint-44000")
parser.add_argument('--use_memory', action="store_true")
parser.add_argument('--memory_file', type=str)
parser.add_argument('--model_name', type=str, help='Model architecture to train', default="openai/whisper-large-v2")
parser.add_argument('--hypo_file', type=str, help='Where to write the hypo')
parser.add_argument('--num_beams', type=int, default=4)
#parser.add_argument('--load_other_base_model', type=str, help='Load other baseline weights for a memory model')
parser.add_argument('--load_adapter_model', type=str, help='Load adapter weights for baseline weights', default="./saves/model_bw/checkpoint-290")

args = parser.parse_args()

if args.hypo_file is None:
    args.hypo_file = f"hypos/hypo_{args.model_path.replace('/','_')}.txt"

print(args)

outputfile = args.hypo_file
if os.path.isfile(outputfile):
    print("Output file already exists, continue?")
    breakpoint()

if len(args.segfiles) == 1:
    dataset = MyDataset(args.segfiles[0], test=True)
else:
    dataset = ConcatDataset([MyDataset(s, test=True) for s in args.segfiles])

if not args.use_memory:
    model_class = WhisperForConditionalGeneration
else:
    model_class = WhisperForConditionalGenerationMemory

    if args.memory_file is not None:
        memory_words = [line.strip() for line in open(args.memory_file)]

        memory = processor.tokenizer(memory_words, return_tensors="pt", padding=True)
        memory["input_ids"] = memory["input_ids"][:,4:].to(device)
        memory["attention_mask"] = memory["attention_mask"][:,4:].to(device)
    else:
        memory = None
    
model = model_class.from_pretrained(args.model_path, torch_dtype="auto", device_map="cuda")

if args.load_adapter_model is not None:
    model = PeftModel.from_pretrained(model, args.load_adapter_model)

tokenizer = WhisperTokenizerFast.from_pretrained(args.model_name)
tokenizer.set_prefix_tokens(language="english", task="transcribe")

processor = WhisperProcessor.from_pretrained(args.model_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer, return_ids=True)

batch_size = 1

forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

with open(outputfile, "w") as f:
    for i in tqdm(range(0,len(dataset),batch_size)):
        data = data_collator([dataset[j] for j in range(i,min(len(dataset),i+batch_size))],inference=False)
        ids = data.pop("ids")

        with torch.no_grad():
            input_features = data["input_features"].cuda()
            transcript = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, no_repeat_ngram_size=6, num_beams=args.num_beams, memory=memory)
        transcript = tokenizer.batch_decode(transcript, skip_special_tokens=True)

        for t,id in zip(transcript, ids):
            print(id,t)
            t = t.replace("\n"," ")
            f.write(id+" "+t+"\n")

