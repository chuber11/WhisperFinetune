
from datasets import load_dataset
import sys
import os
import subprocess
from tqdm import tqdm
from glob import glob

def task(path):
    path1, path2 = path
    if not os.path.isfile(path2):
        subprocess.run([
            "ffmpeg", "-y",                    # overwrite output if it exists
            "-i", path1,                       # input file
            "-ar", "16000",                    # resample to 16kHz
            "-ac", "1",
            path2
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

dataset_name = sys.argv[1] if len(sys.argv) >= 2 else "facebook/voxpopuli"
language = sys.argv[2] if len(sys.argv) >= 3 else "en"
split = sys.argv[3] if len(sys.argv) >= 4 else "validation"

prefix = f"{dataset_name.split('/')[-1]}.{language.upper()}.{split}"

if os.path.isfile(f"data/{prefix}.seg.aligned"):
    input(f"{prefix}.seg.aligned already exists. Press enter to ignore")

dataset = load_dataset(dataset_name, language, split=split)

"""from concurrent.futures import ThreadPoolExecutor, as_completed

data = []
for f in glob("/export/data2/chuber/cache/datasets/downloads/extracted/*/*/*.wav"):
    data.append((f,"/export/data2/chuber/2025/voxpopuli/"+f.split("/")[-1][:-len("wav")]+"mp3"))

with ThreadPoolExecutor(max_workers=16) as executor: # For parallel conversion to mp3
    futures = [executor.submit(task, d) for d in data]

    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()

breakpoint()"""

with open(f"data/{prefix}.seg.aligned","w") as seg, open(f"data/{prefix}.cased","w") as cased:
    for sample in tqdm(dataset):
        if "voxpopuli" in dataset_name:
            key1 = "audio_id"
            key2 = "raw_text"
            path1 = sample["audio"]["path"]
            folder = "voxpopuli"
        elif "fleurs" in dataset_name:
            key1 = "id"
            key2 = "raw_transcription"
            path1 = "/".join(sample["path"].split("/")[:-1])+"/"+sample["audio"]["path"]
            folder = "fleurs"
        elif "librispeech" in dataset_name:
            key1 = "id"
            key2 = "text"
            path1 = subprocess.run("find "+"/".join(sample["file"].split("/")[:-1])+"/* -name "+sample["file"].split("/")[-1], capture_output=True, shell=True).stdout.decode().strip()
            folder = "librispeech"
        id = sample[key1]
        path2 = f"/export/data2/chuber/2025/{folder}/{id}.mp3"
        label = sample[key2]
        if not label.strip():
            continue
        #if not os.path.isfile(path1):
        #    breakpoint()
        #task((path1, path2))
        seg.write(f"{id} {path2}"+"\n")
        cased.write(f"{label}"+"\n")

