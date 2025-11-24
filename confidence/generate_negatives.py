
from analyse_diff import get_id_to_nes, get_id_to_label, get_id_to_path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm
import torchaudio
import json
import os

def load_audio(path, target_sr=16000):
    s,r = "/project/asr_systems/LT2022/data/EN/cv14.0/download","/export/data2/chuber/2024/CV/EN"
    path = path.replace(s,r)
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy()

def split_to_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def to_same_prefix_length(data, batch_size):
    data.sort(key=lambda x: len(x["prefix_ids"]))
    res = [[data[0]]]
    for item in data[1:]:
        if len(item["prefix_ids"]) == len(res[-1][0]["prefix_ids"]):
            res[-1].append(item)
        else:
            res.append([item])
    res = [b for r in res for b in split_to_batches(r, batch_size)]
    return res

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype
).to(device)

processor = AutoProcessor.from_pretrained(model_id)
processor.get_decoder_prompt_ids(language="en", task="transcribe") # Changes internal state

segfiles = {}
segfiles["train"] = ["../../WhisperE+Phi2/data/cv.EN.train.seg.aligned", "../data/voxpopuli.EN.train.seg.aligned"]
segfiles["dev"] = ["../../WhisperE+Phi2/data/cv.EN.dev.seg.aligned", "../data/voxpopuli.EN.validation.seg.aligned"]

for set in ["dev","train"]:
    id_to_nes = get_id_to_nes(f"output/{set}2_filtered.txt")
    id_to_label = get_id_to_label(segfiles[set])
    id_to_path = get_id_to_path(segfiles[set])

    data = []
    for id,nes_ in id_to_nes.items():
        label = id_to_label[id]
        path = id_to_path[id]
        for ne in nes_:
            parts = label.split(ne)
            prefix = parts[0].strip()
            suffix = ne.join(parts[1:]).strip()
            ids = processor.tokenizer(prefix, add_special_tokens=True).input_ids[:-1]
            data.append({"id": id, "path": path, "prefix": prefix, "suffix": suffix, "prefix_ids": ids, "ne": ne, "label": label})

    batch_size_max = 4
    num_return_sequences = 5

    if os.path.isfile(f"output_conf/{set}.txt"):
        continue

    with open(f"output_conf/{set}.txt", "w") as f:
        for batch in tqdm(to_same_prefix_length(data, batch_size_max)):
            audio_arrays = [load_audio(s["path"]) for s in batch]

            inputs = processor(
                audio_arrays,
                return_tensors="pt",
                sampling_rate=16000,
            ).to(device, torch_dtype)

            max_len = max(len(s["prefix_ids"]) for s in batch)
            decoder_input_ids = torch.full(
                (len(batch), max_len),
                processor.tokenizer.pad_token_id,
                dtype=torch.long,
            )

            for i, s in enumerate(batch):
                ids = s["prefix_ids"]
                decoder_input_ids[i, :len(ids)] = torch.tensor(ids)

            decoder_input_ids = decoder_input_ids.to(device)

            generated = model.generate(
                language="en",
                input_features=inputs.input_features,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=100,
                num_return_sequences=num_return_sequences,
                num_beams=5,
            )

            transcripts = processor.batch_decode(generated, skip_special_tokens=True)

            for s, transcript in zip(batch, split_to_batches(transcripts, num_return_sequences)):
                s["hypo"] = transcript

                f.write(json.dumps(s)+"\n")

