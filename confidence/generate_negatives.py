
from analyse_diff import get_id_to_nes, get_id_to_label, get_id_to_path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm
import torchaudio
import json
import os
import random
import queue
import threading
import time
from glob import glob

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
    random.seed(21)
    random.shuffle(res)
    return res

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#model_id = "openai/whisper-tiny"
model_id = "openai/whisper-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype
).to(device)

processor = AutoProcessor.from_pretrained(model_id)
processor.get_decoder_prompt_ids(language="en", task="transcribe") # Changes internal state

segfiles = []
for f in glob(f"../data/*EN*.seg.aligned"):
    segfiles.append(f)
for f in glob(f"../data_yodas/*EN*.seg.aligned"):
    segfiles.append(f)
segfiles = [f for f in segfiles if "train" in f or "dev" in f]
segfiles = [f for f in segfiles if "dev" in f]+[f for f in segfiles if not "dev" in f]

for segfile in segfiles:
    labelfile = segfile.replace(".seg.aligned",".cased")
    dataset_name = labelfile.split('/')[-1]
    rarefile = f"output_rare/{dataset_name}_filtered.txt"

    if os.path.isfile(f"output_conf/{dataset_name}.txt"):
        continue
    print(f"Processing {segfile}")

    id_to_label = get_id_to_label([segfile])
    id_to_path = get_id_to_path([segfile])

    data = []
    for line in tqdm(open(rarefile).readlines()):
        line = line.strip().split()
        id = line[0]
        if len(line) < 2:
            continue
        nes_ = line[1].split(";")

        label = id_to_label[id]
        path = id_to_path[id]
        for ne in nes_:
            if not ne:
                continue
            parts = label.split(ne)
            prefix = parts[0].strip()
            suffix = ne.join(parts[1:]).strip()
            prefix_ids = processor.tokenizer(prefix, add_special_tokens=True).input_ids[:-1]
            data.append({"id": id, "path": path, "prefix": prefix, "suffix": suffix, "prefix_ids": prefix_ids, "ne": ne, "label": label})

    batch_size_max = 2
    max_ne_tokens = 8
    num_return_sequences = 1
    num_beams = 4

    batches = to_same_prefix_length(data, batch_size_max)
    q = queue.Queue()

    def load(q, batches):
        for i,batch in enumerate(tqdm(batches)):
            while q.qsize() > 20:
                time.sleep(0.1)
            if False: #q.qsize() != 20:
                print(f"Loaded {i}/{len(batches)} batches, queue size {q.qsize()}")
            
            audio_arrays = [load_audio(s["path"]) for s in batch]
            inputs = processor(audio_arrays,return_tensors="pt",sampling_rate=16000)
            batch[0]["inputs"] = inputs

            q.put(batch)
        q.put(None)

    thread = threading.Thread(target=load, args=(q, batches))
    thread.start()

    with open(f"output_conf/{dataset_name}.txt", "w") as f:
        while True:
            batch = q.get()
            if batch is None:
                break

            try:
                inputs = batch[0]["inputs"].to(device, torch_dtype)
                decoder_input_ids = processor(text=[s["prefix"] for s in batch],return_tensors="pt")["input_ids"][:,:-1].to(device)

                res_dict = model.generate(
                    #inputs.input_features,
                    #forced_decoder_ids=decoder_input_ids,
                    #language="en",
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
                decoder_input_ids2 = []
                indices = []

                for i, (s, transcripts) in enumerate(zip(batch,
                                                        split_to_batches(res_dict["sequences"].tolist(), num_return_sequences))):
                    suffix = " "+s["suffix"]
                    # remove space if suffix starts with punctuation
                    if suffix[1] in [".",",","!","?",";",":"]:
                        suffix = suffix[1:]
                    suffix_ids = processor.tokenizer(suffix, add_special_tokens=True).input_ids[4:]

                    ids = set()
                    for transcript in transcripts:
                        for j in range(decoder_input_ids.shape[1]+1, len(transcript)+1):
                            if transcript[j-1] == processor.tokenizer.eos_token_id:
                                break
                            new_ids = tuple(transcript[:j]+suffix_ids)
                            ids.add(new_ids)

                    for ids_tuple in ids:
                        decoder_input_ids2.append(ids_tuple)
                        encoder_outputs2.append(encoder_outputs[i])
                        indices.append((i,len(ids_tuple)-len(suffix_ids)-decoder_input_ids.shape[1]))

                encoder_outputs2 = torch.stack(encoder_outputs2).to(device)

                # batch decoder_input_ids2 to tensor and create attention mask
                max_len = max([len(ids) for ids in decoder_input_ids2])
                decoder_input_ids = torch.full((len(decoder_input_ids2), max_len), processor.tokenizer.pad_token_id, dtype=torch.long, device=device)
                decoder_attention_mask = torch.zeros(len(decoder_input_ids2), max_len, dtype=torch.long, device=device)
                for i, ids in enumerate(decoder_input_ids2):
                    decoder_input_ids[i, :len(ids)] = torch.tensor(ids,device=device)
                    decoder_attention_mask[i, :len(ids)] = 1

                with torch.no_grad():
                    output = model.forward(
                        encoder_outputs=[encoder_outputs2],
                        decoder_input_ids=decoder_input_ids[:, :-1],
                        decoder_attention_mask=decoder_attention_mask[:, :-1],
                        return_dict=True
                    )

                    log_probs  = torch.nn.functional.log_softmax(output["logits"], dim=-1)
                    token_log_probs = torch.gather(log_probs, 2, decoder_input_ids[:, 1:].unsqueeze(-1))[:,:,0]

                    mask = decoder_input_ids[:, :-1].ne(processor.tokenizer.pad_token_id).to(log_probs.dtype)

                    token_log_probs = token_log_probs * mask
                    seq_scores = (token_log_probs.sum(-1) / mask.sum(-1)).tolist()

                for j in range(len(seq_scores)):
                    idx,leng = indices[j]
                    if not "hypo_to_score" in batch[idx]:
                        batch[idx]["hypo_to_score"] = {}
                        batch[idx]["hypo_to_leng"] = {}
                    hypo_to_score = batch[idx]["hypo_to_score"]
                    hypo_to_leng = batch[idx]["hypo_to_leng"]
                    hypo = processor.decode(decoder_input_ids2[j], skip_special_tokens=True)
                    hypo_to_score[hypo] = seq_scores[j]
                    hypo_to_leng[hypo] = leng

                del batch[0]["inputs"]
                for s in batch:
                    s["hypo_to_score"] = list(sorted(s["hypo_to_score"].items(), key=lambda item: item[1], reverse=True))
                    #for hypo, score in s["hypo_to_score"]:
                    #    print(f"{s['id'][:5]}\t{s['ne']}\t{score}\t{s['hypo_to_leng'][hypo]}\t{hypo}")
                    #    #break
                    del s["hypo_to_leng"]
                    del s["prefix_ids"]
                    f.write(json.dumps(s) + "\n")
            except Exception as e:
                if "KeyboardInterrupt" in str(e):
                    raise e
                print("ERROR processing batch")
            
