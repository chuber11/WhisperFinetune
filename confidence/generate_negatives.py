
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
    import random
    random.seed(21)
    random.shuffle(res)
    return res

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-tiny" #"openai/whisper-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype
).to(device)

processor = AutoProcessor.from_pretrained(model_id)
processor.get_decoder_prompt_ids(language="en", task="transcribe") # Changes internal state

segfiles = {}
segfiles["train"] = ["../../WhisperE+Phi2/data/cv.EN.train.seg.aligned", "../data/voxpopuli.EN.train.seg.aligned"]
segfiles["dev"] = ["../../WhisperE+Phi2/data/cv.EN.dev.seg.aligned", "../data/voxpopuli.EN.validation.seg.aligned"]

for set_ in ["dev","train"]:
    id_to_nes = get_id_to_nes(f"output/{set_}2_filtered.txt")
    id_to_label = get_id_to_label(segfiles[set_])
    id_to_path = get_id_to_path(segfiles[set_])

    data = []
    for id,nes_ in id_to_nes.items():
        label = id_to_label[id]
        path = id_to_path[id]
        for ne in nes_:
            parts = label.split(ne)
            prefix = parts[0].strip()
            suffix = ne.join(parts[1:]).strip()
            prefix_ids = processor.tokenizer(prefix, add_special_tokens=True).input_ids[:-1]
            data.append({"id": id, "path": path, "prefix": prefix, "suffix": suffix, "prefix_ids": prefix_ids, "ne": ne, "label": label})

    batch_size_max = 2
    max_ne_tokens = 5

    num_return_sequences = 5
    num_beams = 5

    if os.path.isfile(f"output_conf/{set_}.txt"):
        continue

    with open(f"output_conf/{set_}.txt", "w") as f:
        for batch in tqdm(to_same_prefix_length(data, batch_size_max)):
            audio_arrays = [load_audio(s["path"]) for s in batch]

            inputs = processor(audio_arrays,return_tensors="pt",sampling_rate=16000).to(device, torch_dtype)

            decoder_input_ids = processor(text=[s["prefix"] for s in batch],return_tensors="pt")["input_ids"].to(device)

            res_dict = model.generate(
                language="en",
                input_features=inputs.input_features,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_ne_tokens,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
                output_hidden_states=True,
                #no_repeat_ngram_size=5,
                num_beams=num_beams,
            )

            encoder_outputs = res_dict["encoder_hidden_states"][-1]

            encoder_outputs2 = []
            sequences = []
            indices = []

            for i, (s, transcripts) in enumerate(zip(batch,
                                                     split_to_batches(processor.batch_decode(res_dict["sequences"], skip_special_tokens=True), num_return_sequences))):
                transcripts = [t[len(s["prefix"]):].strip() for t in transcripts]
                recognized_nes= set()
                for t in transcripts:
                    parts = t.split()
                    for j in range(len(parts)):
                        recognized_nes.add(" ".join(parts[:j+1]))

                for ne in recognized_nes:
                    encoder_outputs2.append(encoder_outputs[i])
                    indices.append(i)
                    seq = ""
                    if s["prefix"].strip() != "":
                        seq += s["prefix"].strip() + " "
                    seq += ne
                    if s["suffix"].strip() != "":
                        seq += " " + s["suffix"].strip()
                    sequences.append(seq.strip())

            encoder_outputs2 = torch.stack(encoder_outputs2).to(device)

            decoder_input = processor(text=sequences,return_tensors="pt",padding=True)
            decoder_input_ids = decoder_input["input_ids"].to(device)
            decoder_attention_mask = decoder_input["attention_mask"].to(device)

            with torch.no_grad():
                output = model.forward(
                    encoder_outputs=[encoder_outputs2],
                    decoder_input_ids=decoder_input_ids[:, :-1],
                    decoder_attention_mask=decoder_attention_mask[:, :-1],
                    return_dict=True
                )

                log_probs  = torch.nn.functional.log_softmax(output["logits"], dim=-1)
                token_log_probs = torch.gather(log_probs, 2, decoder_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

                mask = decoder_input_ids[:, 1:].ne(processor.tokenizer.pad_token_id).to(log_probs.dtype)

            token_log_probs = token_log_probs * mask
            seq_scores = (token_log_probs.sum(-1) / mask.sum(-1)).tolist()

            for j in range(len(seq_scores)):
                idx = indices[j]
                if not "hypo_to_score" in batch[idx]:
                    batch[idx]["hypo_to_score"] = {}
                hypo_to_score = batch[idx]["hypo_to_score"]
                hypo_to_score[sequences[j]] = seq_scores[j]

            for s in batch:
                for hypo, score in s["hypo_to_score"].items():
                    print(f"{s['id'][:5]}\t{s['ne']}\t{score}\t{hypo}")
                f.write(json.dumps(s) + "\n")
            
