
import torch

from data import MyMTDataset as MyDataset, DataCollatorMTSeq2SeqWithPadding, ConcatDataset

from transformers import MBartForConditionalGeneration, AutoTokenizer

from torch.cuda.amp import autocast

from tqdm import tqdm

import sys
from glob import glob
import os

import argparse

from peft import PeftModel

parser = argparse.ArgumentParser()

parser.add_argument('--segfiles', type=str, nargs="+", help='Data used for training', default=["hypos/hypo_openai_whisper-large-v2_cv_filtered_beam4.*.txt", "hypos/hypo_openai_whisper-large-v2_beam4.*.txt"])
parser.add_argument('--decode_only_first_part', action="store_true")
parser.add_argument('--model_path', type=str, help='Path to store the trained model', default="./saves/model_segmenter1/checkpoint-11500")
parser.add_argument('--model_name', type=str, help='Model architecture to train', default="facebook/mbart-large-50")
parser.add_argument('--hypo_file', type=str, help='Where to write the hypo')
parser.add_argument('--num_beams', type=int, default=4)
parser.add_argument('--load_adapter_model', type=str, help='Load adapter weights for baseline weights', default=None)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--no_write_at_end', action="store_true")

args = parser.parse_args()

if args.hypo_file is None:
    args.hypo_file = f"hypos/hypo_{args.model_path.replace('/','_')}.txt"

print(args)

outputfile = args.hypo_file
if os.path.isfile(outputfile) and os.path.getsize(outputfile) > 0:
    print(f"Output {outputfile} file already exists, exiting!")
    sys.exit()

if len(args.segfiles) == 1:
    dataset = MyDataset(args.segfiles[0], test=True, dev=args.decode_only_first_part)
else:
    dataset = ConcatDataset([MyDataset(s, test=True, dev=args.decode_only_first_part) for s in args.segfiles])

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

data_collator = DataCollatorMTSeq2SeqWithPadding(tokenizer=tokenizer, return_ids=True)

model_class = MBartForConditionalGeneration
    
model = model_class.from_pretrained(args.model_path, torch_dtype="auto", device_map="cuda")

if args.load_adapter_model is not None:
    embeddings = glob(args.load_adapter_model+"/embedding.pt")
    if len(embeddings) == 1:
        embedding = torch.load(embeddings[0])
        embedding = {k[len("model."):]:v for k,v in embedding.items()}
        model.load_state_dict(embedding, strict=False)
        print("Loaded other embedding")
    elif len(embeddings) > 0:
        breakpoint()

if args.load_adapter_model is not None:
    model = PeftModel.from_pretrained(model, args.load_adapter_model)
    model = model.merge_and_unload()

batch_size = args.batch_size

if not args.no_write_at_end:
    outputs = []
else:
    f = open(outputfile, "w")

for i in tqdm(range(0,len(dataset),batch_size)):
    data = data_collator([dataset[j] for j in range(i,min(len(dataset),i+batch_size))],inference=False)
    ids = data.pop("ids")

    with torch.no_grad():
        for k,v in data.items():
            data[k] = v.cuda()
        model.generation_config.decoder_start_token_id = 250004
        transcript = model.generate(**data, no_repeat_ngram_size=6, num_beams=args.num_beams)
    transcript = tokenizer.batch_decode(transcript, skip_special_tokens=True)

    for t,id in zip(transcript, ids):
        print(id,t)
        t = t.replace("\n"," ")
        if not args.no_write_at_end:
            outputs.append(id+" "+t+"\n")
        else:
            f.write(id+" "+t+"\n")

if not args.no_write_at_end:
    with open(outputfile, "w") as f:
        for o in outputs:
            f.write(o)

