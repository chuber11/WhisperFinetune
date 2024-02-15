
import torch
import torchaudio
import soundfile as sf
#import librosa
from torch.utils.data import Dataset

from glob import glob

from dataclasses import dataclass
from transformers import Wav2Vec2Processor
from typing import Any, Dict, List, Union, Optional

from datasets import load_metric

import math
import random

import re
import os
from tqdm import tqdm

def replace_except_specified_chars(text):
    # This pattern matches any character that is NOT a-z, A-Z, äöüÄÖÜß
    pattern = r'[^a-zA-ZäöüÄÖÜß]'
    # Replace these matched characters with a space
    result = re.sub(r'\s+', ' ', re.sub(pattern, ' ', text))
    return result

class MyDataset(Dataset):
    def __init__(self, dev=False, segfiles=None, replace=None, max_len=2000, augment=False, memory=False):
        if segfiles is None:
            #segfiles = "data/*.train.seg.aligned"
            segfiles = "../WhisperE+Phi2/data/cv.*.train.seg.aligned"

        if dev:
            segfiles = segfiles.replace("train","dev")

        if augment:
            segfiles = segfiles.replace("data","data_augment")

        if replace is None:
            replace = [("/project/asr_systems/LT2021/EN/data","/export/data2/chuber/ASR/data/EN")]

        self.ids = []
        self.audio_paths = []
        self.timestamps = []
        self.labels = []
        for segfile in glob(segfiles):
            print(segfile)
            labelfile = ".".join(segfile.split(".")[:-2])+".cased"
            for line, line2 in zip(open(segfile),open(labelfile)):
                line = line.strip().split()
                self.ids.append(line[0])
                audio_path = line[1]
                for r in replace:
                    audio_path = audio_path.replace(r[0],r[1])
                self.audio_paths.append(audio_path)
                if len(line) == 2:
                    self.timestamps.append(None)
                elif len(line) == 4:
                    self.timestamps.append((float(line[2]),float(line[3])))
                else:
                    raise RuntimeError
                lang = "<|en|>"
                if "DE" in segfile:
                    lang = "<|de|>"
                self.labels.append(lang+line2.strip())

                #if len(self.audio_paths) >= 16*3:
                #    break

        random.seed(42)

        combined_lists = list(zip(self.ids, self.audio_paths, self.timestamps, self.labels))
        random.shuffle(combined_lists)
        self.ids, self.audio_paths, self.timestamps, self.labels = zip(*combined_lists)

        self.len = len(self.audio_paths)
        if dev:
            self.len = min(max_len,self.len)

        self.memory = memory

        if memory:
            new_words_list = f"new_words_list_{segfiles.replace('/','_').replace('train','').replace('dev','')}.pt"

            if not dev and not os.path.isfile(new_words_list):
                from collections import Counter

                occ = Counter()
                for label in tqdm(self.labels):
                    words = replace_except_specified_chars(label[len("<|en|>"):]).split()
                    for word in words:
                        word = label[:len("<|en|>")]+word
                        occ[word] += 1

                """for i in range(1,20):
                    tmp = [k for k,v in occ.items() if v==i]
                    print(i,len(tmp),tmp[:20])"""

                new_words = set(k for k,v in occ.items() if v<=5)

                torch.save(new_words, new_words_list)

            self.new_words = list(torch.load(new_words_list))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.timestamps[idx] is not None:
            start, end = self.timestamps[idx]
            audio, sr = sf.read(self.audio_paths[idx], start=int(16000*start),stop=int(16000*end))
            #audio, sr  = torchaudio.load(self.audio_paths[idx], frame_offset=int(16000*start),num_frames=int(16000*(end-start)))
            #audio = audio[0]
        else:
            #audio, sr = torchaudio.load(self.audio_paths[idx])
            audio, sr = sf.read(self.audio_paths[idx])
            #audio, sr = librosa.load(self.audio_paths[idx])
        #sample = {"audio":audio[0].numpy(),"labels":self.labels[idx]}
        sample = {"audio":audio,"labels":self.labels[idx], "id":self.ids[idx]}

        if self.memory:
            label = sample["labels"]
            label_words = replace_except_specified_chars(label[len("<|en|>"):]).split()
            memory_words = set(label[:len("<|en|>")]+word for word in label_words if label[:len("<|en|>")]+word in self.new_words)

            sample["memory_words"] = memory_words
            sample["memory_word_dummys"] = random.sample(self.new_words,4)

        return sample

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    tokenizer: Any
    return_ids: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]],                 inference=False) -> Dict[str, torch.Tensor]:

        audio = torch.cat([self.processor(item["audio"], sampling_rate=16000,                     return_tensors="pt").input_features for item in features], dim=0)
        text_labels = self.tokenizer([feature["labels"] for feature in features], return_tensors="pt", padding=True)

        text_labels["input_ids"] = torch.cat([
            text_labels["input_ids"][:,:1],
            text_labels["input_ids"][:,3:4],
            text_labels["input_ids"][:,1:3],
            text_labels["input_ids"][:,4:]
            ],1)

        input_ids = text_labels["input_ids"][:,:-1]

        mask = text_labels["attention_mask"][:,1:]
        labels = text_labels["input_ids"][:,1:].clone()
        labels[mask.eq(0)] = -100

        batch = {"input_features": audio, "decoder_input_ids":input_ids, "labels":labels}
        if self.return_ids:
            batch["ids"] = [item["id"] for item in features]

        if "memory_words" in features[0]:
            memory_length_max = 100

            memory_words = [word for feature in features for word in feature["memory_words"]]+[word for feature in features for word in feature["memory_word_dummys"]]
            memory_words = memory_words[:memory_length_max]

            memory = self.tokenizer(memory_words, return_tensors="pt", padding=True)
            memory["input_ids"] = memory["input_ids"][:,3:]
            memory["attention_mask"] = memory["attention_mask"][:,3:]

            memory_labels = torch.zeros_like(labels)
            index = 0
            for index2, feature in enumerate(features):
                for word in feature["memory_words"]:
                    lang, word = word[:len("<|en|>")], word[len("<|en|>"):]

                    ids = labels[index2][3:]
                    mask = ids.ne(-100)
                    ids = ids[mask][:-1]

                    tokens = [self.tokenizer.decode(i) for i in ids]
                    label = feature["labels"][len("<|en|>"):]

                    for start in range(len(tokens)):
                        label = label[len(tokens[start]):]
                        if not word in label:
                            label = tokens[start]+label
                            break
                    for end in range(len(tokens)-1,-1,-1):
                        label = label[:-len(tokens[end])]
                        if not word in label:
                            label = label+tokens[end]
                            break

                    if word not in self.tokenizer.decode(labels[index2,start+3:end+4]):
                        print("WARNING: Memory entry might be corrupted")

                    index += 1
                    memory_labels[index2,start+3:end+4] = index

            #print(memory_labels.eq(0).sum(),memory_labels.ne(0).sum())

            batch["memory"] = memory
            batch["memory_labels"] = memory_labels

        return batch

def compute_metrics(pred):
    #print(pred)
    breakpoint()
    return {}

    breakpoint()
    loss = pred.loss
    ppl = math.exp(loss.sum()/loss.shape[0])
    return {"ppl": ppl}

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"ppl":ppl, "wer": wer}
