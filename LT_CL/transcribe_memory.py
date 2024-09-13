
from glob import glob
from tqdm import tqdm
import requests
import os
import sys
import time
import json
import subprocess
from collections import Counter

def convert_mp3_to_wav_bytes(mp3_file_path):
    # Command to call ffmpeg and convert mp3 to wav with specified parameters
    command = [
        'ffmpeg', 
        '-i', mp3_file_path,        # Input file
        '-f', 'wav',                # Output format
        '-acodec', 'pcm_s16le',     # Audio codec
        '-ar', '16000',             # Audio sampling rate
        '-ac', '1',                 # Number of audio channels
        'pipe:1'                    # Output to stdout
    ]
    
    # Run the command and capture the output bytes
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Check for errors
    if result.returncode != 0:
        raise Exception(f"ffmpeg error: {result.stderr.decode('utf-8')}")
    
    return result.stdout[100:] # remove header

if __name__ == "__main__":
    user = sys.argv[1]
    server = "http://192.168.0.60:4999/asr/infer/en+de,en+de"
    server_baseline = "http://192.168.0.60:5008/asr/infer/en+de,en+de"
    num_train_before_dev = 9
    individual_words = True

    while True:
        try:
            res = requests.post(server)
        except requests.exceptions.ConnectionError:
            print("Model not running, waiting...")
            time.sleep(10)
        else:
            break

    recognized_words = {}
    for split in ["train","dev"]:
        c = Counter()
        for memoryfile in glob(f"data_processed_{user}/*.{split}.memory"):
            for line in open(memoryfile):
                for word in line.strip().split("|"):
                    if not word:
                        continue
                    c[word] += 1
        recognized_words[split] = c

    for segfile in tqdm(sorted(glob(f"data_to_process_{user}/*.seg.aligned"))):
        id = segfile[:-len("seg.aligned")]
        mp3 = id + "mp3"

        if not os.path.isfile(mp3):
            continue

        wav_bytes = convert_mp3_to_wav_bytes(mp3)
        memoryfile = id + "memory"
        memory_words = [line.strip() for line in open(memoryfile) if line.strip()]
        if individual_words:
            memory_words = list(set(word for phrase in memory_words for word in phrase.split() if word.strip()))
        print(f"{memory_words = }")

        outfiles = {}
        for typ in ["seg.aligned","hypo","memory"]:
            for split in ["train","dev"]:
                outfiles[(typ,split)] = f"{id}{split}.{typ}"

        all_s, all_h, all_m = [], [], []

        for line in open(segfile):
            _, _, start, end = line.strip().split()
            wav_ = wav_bytes[int(32000*float(start)):int(32000*float(end))]
            if len(wav_) % 2 != 0:
                wav_ = wav_[:-1]

            res = requests.post(server, files={"pcm_s16le":wav_, "memory":json.dumps(memory_words), "prefix":"", "user":user, "num_beams": 4})
            print(res.text)
            hypo = res.json()["hypo"]

            res = requests.post(server_baseline, files={"pcm_s16le":wav_, "prefix":"", "num_beams": 4})
            hypo_baseline = res.json()["hypo"]

            if any(word in hypo and word not in hypo_baseline for word in memory_words) or any(word in hypo and word not in hypo_baseline for c in recognized_words.values() for word in c):
                all_s.append(line)
                all_h.append(hypo+"\n")
                words = set()
                for word in memory_words:
                    if word in hypo:
                        words.add(word)
                for c in recognized_words.values():
                    for word in c:
                        if word in hypo:
                            words.add(word)
                all_m.append(sorted(list(words)))

        if len(all_s) > 0:
            for f in outfiles.values(): # Create all files
                open(f,"a").write("")

            for s,h,m in zip(all_s, all_h, all_m):
                split = "train"
                if all(recognized_words["train"][word] >= num_train_before_dev for word in m) and not all(word in recognized_words["dev"] for word in m):
                    split = "dev"
                open(outfiles[("seg.aligned",split)],"a").write(s)
                open(outfiles[("hypo",split)],"a").write(h)
                open(outfiles[("memory",split)],"a").write("|".join(m)+"\n")
                for word in m:
                    recognized_words[split][word] += 1

