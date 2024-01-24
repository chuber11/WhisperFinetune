
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

class MyDataset(Dataset):
    def __init__(self, dev=False, segfiles=None, replace=None, max_len=2000, augment=False):
        if segfiles is None:
            segfiles = "data/*.train.seg.aligned"
            #segfiles = "/project/OML/chuber/2022/NMTGMinor/exp/ASR-NW/data/orig_en_cased/cv.train.seg.aligned"
            #segfiles = "/project/OML/chuber/2022/NMTGMinor/exp/ASR-NW/data/orig_en_cased/*.train.seg.aligned"

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
                self.labels.append(line2.strip())

                #if len(self.audio_paths) >= 16*3:
                #    break

        random.seed(42)

        combined_lists = list(zip(self.ids, self.audio_paths, self.timestamps, self.labels))
        random.shuffle(combined_lists)
        self.ids, self.audio_paths, self.timestamps, self.labels = zip(*combined_lists)

        self.len = len(self.audio_paths)
        if dev:
            self.len = min(max_len,self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.timestamps[idx] is not None: # TODO: only load relevant audio
            start, end = self.timestamps[idx]
            audio, sr = sf.read(self.audio_paths[idx], start=int(16000*start),stop=int(16000*end))
            #audio, sr  = torchaudio.load(self.audio_paths[idx], frame_offset=int(16000*start),num_frames=int(16000*(end-start)))
            #audio = audio[0]
        else:
            #audio, sr = torchaudio.load(self.audio_paths[idx])
            audio, sr = sf.read(self.audio_paths[idx])
            #audio, sr = librosa.load(self.audio_paths[idx])
        #sample = {"audio":audio[0].numpy(),"labels":self.labels[idx]}
        sample = {"input_values":audio,"labels":self.labels[idx], "id":self.ids[idx]}
        return sample

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        label_features = [{"input_ids": self.processor.tokenizer(feature["labels"])["input_ids"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

    loss = pred.loss
    ppl = math.exp(loss.sum()/loss.shape[0])

    return {"ppl": ppl, "wer": wer}

