
import requests
import random
import os
import sys
from tqdm import tqdm
#from qwen_asr import parse_asr_output
import base64
from multiprocessing.pool import ThreadPool as Pool

def get_prompt(audio_path: str, memory_words):
    b64_audio = base64.b64encode(open(audio_path,"rb").read()).decode("ascii")
    audio = {"data": b64_audio, "format": "wav"}
    if model == "qwen3asr":
        if not memory_words:
            context = ""
        else:
            context = separation.join(memory_words)
        return [
            {
                "role": "system",
                "content": context,
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": audio},
                ]
            }
        ]
    elif model == "qwen3omni":
        text = f"Transcribe all speech in the audio completely and accurately. Do not omit any words."
        if memory_words:
            text += f" Note: the following word(s) may appear in the audio and should be spelled as given if heard: {separation.join(memory_words)}."
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": audio},
                    {"type": "text", "text": text},
                ]
            }
        ]

def transcribe(conversations):
    #conda deactivate
    #conda activate qwen3-asr
    #qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 7680
    #vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 7680 --tensor-parallel-size 2

    if model == "qwen3asr":
        url = "http://localhost:7681/v1/chat/completions"
    elif model == "qwen3omni":
        url = "http://localhost:7680/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    def send_request(conversation):
        data = {
            "messages": conversation,
            "temperature": 0.0,
            "max_tokens": 256
        }

        response = requests.post(url, headers=headers, json=data, timeout=300)
        response.raise_for_status()

        content = response.json()['choices'][0]['message']['content']
        
        if model == "qwen3asr":
            if not "<asr_text>" in content:
                print("WARNING: No <asr_text> decoded!")
                text = content
            else:
                language, text = content.split("<asr_text>")
        elif model == "qwen3omni":
            text = content

        return text

    results = []
    with Pool() as pool:
        for r in pool.imap(send_request, conversations):
            results.append(r)

    return results

def split_into_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

n1 = int(sys.argv[1])
n2 = int(sys.argv[2])

if n1 == 0:
    model="qwen3asr"
elif n1 == 1:
    model="qwen3omni"

if n2 == 0:
    separation = ", "
elif n2 == 1:
    separation = " "

for set_ in ["earnings","librispeech_asr.CLEAN","librispeech_asr.OTHER","yodas"]:
    if "librispeech" in set_:
        segfile = f"/export/data2/chuber/2024/Whisper+Memory/data_filtered_test/{set_}_memory.EN.test.seg.aligned"
    elif "yodas" in set_:
        segfile = f"/export/data2/chuber/2024/Whisper+Memory/data_filtered_test/yodas_memory.EN.test.seg.aligned"
    else:
        segfile = "/home/chuber/2025/data/earnings21/earnings_memory.EN.test.seg.aligned"

    memoryfile = segfile.replace("seg.aligned","words")

    new_words = [(line2.strip().split()[0],line.strip().split("|")) for line,line2 in zip(open(memoryfile),open(segfile))]
    id2newwords = {id:words for id,words in new_words}

    for distractors in [0,10,100,250,-250,-100,-10,-1]:
        if model == "qwen3asr":
            if separation == ", ":
                outfile = f"hypos_memory_diss/saves_model_qwen3asr1p7B_random.EN.data_filtered_test_{set_}_memory.EN.test.allwords.{distractors}.0.hyp"
            else:
                outfile = f"hypos_memory_diss/saves_model_qwen3asr1p7B_random2.EN.data_filtered_test_{set_}_memory.EN.test.allwords.{distractors}.0.hyp"
        elif model == "qwen3omni":
            if separation == ", ":
                outfile = f"hypos_memory_diss/saves_model_qwen3omni30B_random.EN.data_filtered_test_{set_}_memory.EN.test.allwords.{distractors}.0.hyp"
            else:
                outfile = f"hypos_memory_diss/saves_model_qwen3omni30B_random2.EN.data_filtered_test_{set_}_memory.EN.test.allwords.{distractors}.0.hyp"
        else:
            raise NotImplementedError

        if os.path.exists(outfile):
            print(f"{outfile} exists, skip")
            continue

        print(f"Processing set {set_} with distractors {distractors}")

        random.seed(42)

        data = []
        for line in tqdm(open(segfile).readlines()):
            id, audio_path = line.strip().split()

            memory_words = [w for w in id2newwords[id]]
            l_mem = len(memory_words)
            if distractors not in [0,-1]:
                num = 0
                for _,words in new_words:
                    for word in words:
                        if num >= abs(distractors):
                            break
                        if word not in memory_words:
                            memory_words.append(word)
                            num += 1
                    if num >= abs(distractors):
                        break
            if distractors < 0:
                memory_words = memory_words[l_mem:]
            random.shuffle(memory_words)

            conversation = get_prompt(audio_path, memory_words)
            data.append((id, conversation))
            
        batch_size = 64

        batches = list(split_into_batches(data, batch_size))

        with open(outfile,"w") as f:
            for batch in tqdm(batches):
                ids, conversations = zip(*batch)
                results = transcribe(conversations)
                for id, res in zip(ids, results):
                    f.write(f"{id} {res}\n")

