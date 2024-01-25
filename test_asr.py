
import torch

from data import MyDataset, DataCollatorSpeechSeq2SeqWithPadding

from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast

from torch.cuda.amp import autocast

from tqdm import tqdm

import sys
from glob import glob
import os

path = sys.argv[1] if len(sys.argv) > 1 else ""
segfile = sys.argv[2] if len(sys.argv) > 2 else "data/*.test.seg.aligned"

print("Using path",path)

model = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, device_map="cuda")

#dataset = MyDataset(segfiles=segfile, dev=True) # Only decode first part of testset
dataset = MyDataset(segfiles=segfile) # Decode whole testset

model_name = "openai/whisper-large-v3"
#model_name = "openai/whisper-medium"

tokenizer = WhisperTokenizerFast.from_pretrained(model_name)
tokenizer.set_prefix_tokens(language="german", task="transcribe")

processor = WhisperProcessor.from_pretrained(model_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer, return_ids=True)

batch_size = 64

outputfile = f"hypos/hypo_{path.replace('/','_')}.txt"
if os.path.isfile(outputfile):
    print("Output file already exists, continue?")
    breakpoint()

forced_decoder_ids = processor.get_decoder_prompt_ids(language="german", task="transcribe")

with open(outputfile, "w") as f:
    for i in tqdm(range(0,len(dataset),batch_size)):
        data = data_collator([dataset[j] for j in range(i,min(len(dataset),i+batch_size))],inference=False)
        ids = data.pop("ids")

        with torch.no_grad():
            input_features = data["input_features"].cuda()
            transcript = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, no_repeat_ngram_size=6)
        transcript = tokenizer.batch_decode(transcript, skip_special_tokens=True)

        for t,id in zip(transcript, ids):
            print(id,t)
            t = t.replace("\n"," ")
            f.write(id+" "+t+"\n")

