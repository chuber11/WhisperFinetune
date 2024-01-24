
import torch

from data import MyDataset

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor

from torch.cuda.amp import autocast

from tqdm import tqdm

import sys
from glob import glob
import os

path = sys.argv[1] if len(sys.argv) > 1 else ""
segfile = sys.argv[2] if len(sys.argv) > 2 else "data/*.test.seg.aligned"
device = "cuda"

print("Using path",path)

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = Wav2Vec2ForCTC.from_pretrained(path).to(device)

#dataset = MyDataset(segfiles=segfile, dev=True) # Only decode first part of testset
dataset = MyDataset(segfiles=segfile) # Decode whole testset

outputfile = f"hypos/hypo_{path.replace('/','_')}.txt"
if os.path.isfile(outputfile):
    print("Output file already exists, continue?")
    breakpoint()

with open(outputfile, "w") as f:
    for i in tqdm(range(0,len(dataset))):
        data = dataset[i]
        input_values = torch.as_tensor(data["input_values"], dtype=torch.float32).to(device).unsqueeze(0)
        id = data["id"]

        with torch.no_grad():
            print(input_values.shape)
            logits = model(input_values).logits
            print(logits.shape)

        pred_ids = torch.argmax(logits, dim=-1)
        print(pred_ids)
        t = processor.batch_decode(pred_ids)[0]

        print(id,t)
        t = t.replace("\n"," ")
        f.write(id+" "+t+"\n")

        break

