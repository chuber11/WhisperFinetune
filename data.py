
import torch
import torchaudio
import soundfile as sf
#import librosa
from torch.utils.data import Dataset

from glob import glob

from dataclasses import dataclass
from transformers import Wav2Vec2Processor, WhisperTokenizerFast, WhisperProcessor
from typing import Any, Dict, List, Union, Optional

import math
import random
import json

import re
import os
from tqdm import tqdm

from torch.utils.data import DataLoader

def replace_except_specified_chars(text):
    # This pattern matches any character that is NOT a-z, A-Z, äöüÄÖÜß
    pattern = r'[^a-zA-ZäöüÄÖÜß\'–—’&]'
    # Replace these matched characters with a space
    result = re.sub(r'\s+', ' ', re.sub(pattern, ' ', text))
    return result

class MyDataset(Dataset):
    def __init__(self, segfiles, dev=False, replace=None, max_len=2000, memory=False, test=False, confidence=False):
        if replace is None:
            replace = [("/project/asr_systems/LT2022/data/DE/cv14.0/download","/export/data2/chuber/2024/CV/DE"),
                       ("/project/asr_systems/LT2022/data/EN/cv14.0/download","/export/data2/chuber/2024/CV/EN")]

        self.ids = []
        self.audio_paths = []
        self.timestamps = []
        self.labels = []

        self.memory = memory and not test
        self.confidence = confidence

        all_segfiles = glob(segfiles)
        if len(all_segfiles) == 0 and os.path.isfile(segfiles):
            all_segfiles = [segfiles]

        for segfile in all_segfiles:
            print(segfile)
            if confidence:
               self.load_confidence(segfile)
               continue

            labelfile = ".".join(segfile.split(".")[:-2])+".cased"
            if not os.path.isfile(labelfile):
                labelfile = labelfile[:-len(".cased")]+".ref"
                if not os.path.isfile(labelfile):
                    labelfile = labelfile[:-len(".ref")]+".hypo"
            if not os.path.isfile(labelfile):
                if not test:
                    raise FileNotFoundError
                labelfile = None

            if labelfile is not None:
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
                    prefix = "" if not self.memory else " "
                    self.labels.append(lang+prefix+line2.strip())
            else:
                for line in open(segfile):
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
                    self.labels.append(None)

        if len(all_segfiles) == 0:
            print(segfiles)
            input("Dataset does not contain segfiles. Press enter to ignore: ")

        random.seed(42)

        if not test and len(self.audio_paths) > 0:
            combined_lists = list(zip(self.ids, self.audio_paths, self.timestamps, self.labels))
            random.shuffle(combined_lists)
            self.ids, self.audio_paths, self.timestamps, self.labels = zip(*combined_lists)

        self.len = len(self.audio_paths)
        if dev:
            self.len = min(max_len,self.len)

        if self.memory:
            new_words_list = f"new_words_list_{segfiles.replace('/','_').replace('train','').replace('dev','')}.pt"

            if not dev and not os.path.isfile(new_words_list):
                from collections import Counter

                """occ = Counter()
                for label in tqdm(self.labels):
                    for c in label.strip():
                        occ[c] += 1
                print(sorted(occ.items(),key=lambda x:-x[1]))
                breakpoint()"""

                occ = Counter()
                for label in tqdm(self.labels):
                    words = replace_except_specified_chars(label[len("<|en|>"):]).split()
                    for word in words:
                        occ[word] += 1

                """for i in range(1,20):
                    tmp = [k for k,v in occ.items() if v==i]
                    print(i,len(tmp),tmp[:20])"""

                new_words = set(k for k,v in occ.items()) # if v<=10)

                torch.save(new_words, new_words_list)

            self.new_words = list(torch.load(new_words_list))

        print("Dataset has length",len(self))

    def load_confidence(self, segfile):
        for line in open(segfile):
            data = json.loads(line.strip())

            if not "label" in data:
                hypo = data["hypo"]
                self.ids.append(data["id"])
                self.audio_paths.append(data["path"])
                self.timestamps.append((data["start"],data["end"]))
                self.labels.append(hypo)
                continue

            label = "<|en|>"+data["label"]
            nes = data["nes"]

            self.ids.append(data["id"])
            self.audio_paths.append(data["path"])
            self.timestamps.append(None)
            self.labels.append([label, nes, 0])

            if any(ne["ne_hypo"] != ne["ne"] for ne in nes):
                for ne in nes:
                    ne_ = ne["ne"]
                    ne_hypo_ = ne["ne_hypo"]
                    label = label.replace(ne_, ne_hypo_)
                if len(label) > 1.5 * len(data["label"]):
                    continue
                self.ids.append(data["id"])
                self.audio_paths.append(data["path"])
                self.timestamps.append(None)
                self.labels.append([label, nes, 1])

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
            try:
                audio, sr = sf.read(self.audio_paths[idx])
            except:
                print("ERROR: Audiofile does not exist:",self.audio_paths[idx])
                raise
            #audio, sr = librosa.load(self.audio_paths[idx])
        #sample = {"audio":audio[0].numpy(),"labels":self.labels[idx]}
        sample = {"audio":audio,"labels":self.labels[idx], "id":self.ids[idx]}

        if self.memory:
            label = sample["labels"]
            label_words = replace_except_specified_chars(label[len("<|en|>"):]).split()
            memory_words = set(word for word in label_words if word in self.new_words)

            sample["memory_words"] = memory_words
            sample["memory_word_dummys"] = random.sample(self.new_words,4)

        if self.confidence:
            if "labels" in sample and isinstance(sample["labels"], list):
                sample["labels_parts"] = sample["labels"]
                sample["labels"] = sample["labels"][0]

        return sample

class ConcatDataset(Dataset):
    def __init__(self, datasets, factors=None):
        self.datasets = datasets
        self.factors = factors is not None

        self.cumulative_lengths = self._compute_cumulative_lengths(datasets, factors)

        self.order = random.sample(range(len(self)), len(self))

        print("ConcatDataset has length",len(self))

    def _compute_cumulative_lengths(self, datasets, factors):
        cumulative_lengths = []
        total_length = 0

        if len(datasets) == 0:
            raise RuntimeError("At least one dataset has to be provided!")

        if factors is None:
            for dataset in datasets:
                total_length += len(dataset)
                cumulative_lengths.append(total_length)
        else:
            if len(datasets) != len(factors):
                raise RuntimeError(f"The number of given datasets and the number of dataset factors do not match: {len(datasets)}, {len(factors)}")

            print("Dataset weighting factors:",factors)

            for dataset, factor in zip(datasets,factors):
                total_length += factor*len(dataset)
                cumulative_lengths.append(total_length)

        return cumulative_lengths

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        idx = self.order[idx]

        # Determine which dataset the index falls into and the index within that dataset
        dataset_index = next(i for i, cumulative_length in enumerate(self.cumulative_lengths) if idx < cumulative_length)
        if dataset_index == 0:
            index_within_dataset = idx
        else:
            index_within_dataset = idx - self.cumulative_lengths[dataset_index - 1]

        dataset = self.datasets[dataset_index]

        if self.factors:
            index_within_dataset %= len(dataset)

        return dataset[index_within_dataset]

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    tokenizer: Any
    return_ids: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]], inference=False) -> Dict[str, torch.Tensor]:

        audio = torch.cat([self.processor(item["audio"], sampling_rate=16000, return_tensors="pt").input_features for item in features], dim=0)

        batch = {"input_features": audio}

        has_memory = "memory_words" in features[0]
        has_confidence = "labels_parts" in features[0]

        if has_memory:
            memory_length_max = 200
            avg_words_per_utterance = 3

            info = [(index2, feature, word) for index2, feature in enumerate(features) for word in feature["memory_words"]]
            info = random.sample(info, avg_words_per_utterance*len(features))

            memory_words = [" "+word for index2, feature, word in info]+[" "+word for feature in features for word in feature["memory_word_dummys"]]
            memory_words = memory_words[:memory_length_max]
            for feature in features:
                labels = feature["labels"]
                for i,memory_word in enumerate(memory_words):
                    if memory_word not in labels:
                        continue
                    labels = labels.replace(memory_word,f"<|memory_{i}|>")
                feature["labels"] = labels

            pattern = r'(?:<\|memory_\d+\|>){2,}'

            for feature in features:
                #print("Labels with memory token:", feature["labels"])
                for match in re.findall(pattern, feature["labels"][len("<|en|>"):]):
                    ids = list(map(int, re.findall(r'<\|memory_(\d+)\|>', match)))
                    if random.random() < 0.5: # 50% chance to merge
                        #print("Match",match,ids)
                        feature["labels"] = feature["labels"].replace(match, f"<|memory_{len(memory_words)}|>")
                        memory_words.append("".join(memory_words[id] for id in ids))

            memory = self.tokenizer(memory_words, return_tensors="pt", padding=True)
            memory["input_ids"] = memory["input_ids"][:,3:]
            memory["attention_mask"] = memory["attention_mask"][:,3:]
            
            batch["memory"] = memory
        
        if has_memory:
            batch["first_memory_id"] = self.tokenizer("<|memory_0|>")["input_ids"][-2]

        if features[0]["labels"] is not None:
            text_labels = self.tokenizer([feature["labels"] for feature in features], return_tensors="pt", padding=True)

            #print(1,self.processor.tokenizer.decode(text_labels["input_ids"][0]))

            text_labels["input_ids"] = torch.cat([
                text_labels["input_ids"][:,:1],
                text_labels["input_ids"][:,4:5],
                text_labels["input_ids"][:,2:4],
                text_labels["input_ids"][:,5:]
                ],1)
            text_labels["attention_mask"] = torch.cat([
                text_labels["attention_mask"][:,:1],
                text_labels["attention_mask"][:,4:5],
                text_labels["attention_mask"][:,2:4],
                text_labels["attention_mask"][:,5:]
                ],1)

            # has to be: START LANG TRANSCRIBE NO_TIMESTAMPS
            #print(2,self.processor.tokenizer.decode(text_labels["input_ids"][0]))
            #exit()

            input_ids = text_labels["input_ids"][:,:-1]

            mask = text_labels["attention_mask"][:,1:]
            labels = text_labels["input_ids"][:,1:].clone()
            labels[mask.eq(0)] = -100

            batch.update({"decoder_input_ids":input_ids, "labels":labels})

        if has_confidence:
            confidence_label = torch.zeros_like(batch["labels"])
            for i,feature in enumerate(features):
                label, nes, number = feature["labels_parts"]
                if not nes:
                    continue
                
                for ne in nes:
                    ne_ = ne["ne"]
                    ne_hypo_ = ne["ne_hypo"] if number > 0 else ne["ne"]
                    confidence = ne_ == ne_hypo_

                    for h in [" "+ne_hypo_, ne_hypo_]:
                        ids = self.tokenizer(" "+ne_hypo_, add_special_tokens=False).input_ids
                        for j in range(batch["labels"].shape[1]-len(ids)):
                            if torch.all(batch["labels"][i,j:j+len(ids)] == torch.tensor(ids)):
                                confidence_label[i,j:j+len(ids)] = 1 if confidence else 2

            confidence_label[mask.eq(0)] = -100
            batch["confidence_labels"] = confidence_label

        if self.return_ids:
            batch["ids"] = [item["id"] for item in features]
            
        return batch

def compute_metrics(pred):
    #print(pred)
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

class MyMTDataset(Dataset):
    def __init__(self, segfiles, dev=False, test=False, max_len=2000):
        self.test = test

        self.sources = []
        if not test:
            self.targets = []

            for segfile in glob(segfiles):
                print(segfile)

                target = ".".join(segfile.split(".")[:-2])+".cased"
                source = ".".join(segfile.split(".")[:-2])+".tts"

                lang = "en_XX"
                if "DE" in segfile:
                    lang = "de_DE"

                for line, line2 in zip(open(source),open(target)):
                    line = line.strip()
                    line2 = line2.strip()

                    if len(line) > 5*len(line2):
                        continue

                    self.sources.append(line)
                    self.targets.append(line2)
        else:
            self.ids = []

            for segfile in glob(segfiles):
                print(segfile)

                for line in open(segfile):
                    line = line.strip().split()
                    id, src = line[0], " ".join(line[1:])

                    self.ids.append(id)
                    self.sources.append(src)

        #print(sum(len(s) for s in self.sources)/len(self.sources),max(len(s) for s in self.sources))
        #print(sum(len(s) for s in self.targets)/len(self.targets),max(len(s) for s in self.targets))

        self.len = len(self.sources)
        #if dev:
        #    self.len = min(max_len,self.len)
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if not self.test:
            return {"src":self.sources[idx], "tgt":self.targets[idx]}
        else:
            return {"src":self.sources[idx], "id":self.ids[idx]}

@dataclass
class DataCollatorMTSeq2SeqWithPadding:
    tokenizer: Any
    return_ids: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]],                 inference=False) -> Dict[str, torch.Tensor]:
        source = self.tokenizer([f["src"] for f in features],return_tensors="pt",padding=True)

        batch = source

        if "tgt" in features[0]:
            target = self.tokenizer([f["tgt"] for f in features],return_tensors="pt",padding=True)

            input_ids = target["input_ids"][:,:-1]

            mask = target["attention_mask"][:,1:]
            labels = target["input_ids"][:,1:].clone()
            labels[mask.eq(0)] = -100

            batch.update({"decoder_input_ids":input_ids, "decoder_attention_mask":mask, "labels":labels})
        else:
            ids = [f["id"] for f in features]
            batch["ids"] = ids

        return batch

if __name__ == "__main__":

    segfile = "confidence/output_combined/dev.txt"
    dataset = MyDataset(segfile, confidence=True)

    model_name = "openai/whisper-large-v2"

    tokenizer = WhisperTokenizerFast.from_pretrained(model_name)
    #tokenizer.set_prefix_tokens(language="german", task="transcribe")
    tokenizer.set_prefix_tokens(task="transcribe")
    tokenizer.pad_token = tokenizer.eos_token
    processor = WhisperProcessor.from_pretrained(model_name)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=2)

    for batch in dataloader:
        cl = batch["confidence_labels"]
        for cl_ in cl:
            mask = cl_.eq(1) | cl_.eq(2)
            if mask.any():
                print(cl_)

    sys.exit()

    #segfiles = ["../WhisperE+Phi2/data/*.dev.seg.aligned"]
    segfiles = ["data/cv.*.dev.seg.aligned"]

    dataset = ConcatDataset([MyDataset(segfile, memory=True) for segfile in segfiles])

    model_name = "openai/whisper-large-v2"

    tokenizer = WhisperTokenizerFast.from_pretrained(model_name)
    #tokenizer.set_prefix_tokens(language="german", task="transcribe")
    tokenizer.set_prefix_tokens(task="transcribe")
    tokenizer.pad_token = tokenizer.eos_token
    processor = WhisperProcessor.from_pretrained(model_name)

    new_tokens = [f"<|memory_{i}|>" for i in range(200)]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} memory tokens")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=16)

    s = 0
    n = 0
    for batch in dataloader:
        s += batch["labels"].ne(-100).sum()
        n += 1
        print(s/n)

