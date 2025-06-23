
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

dataset = sys.argv[1] if len(sys.argv) >= 2 else "facebook/voxpopuli"
language = sys.argv[2] if len(sys.argv) >= 3 else "en"
split = sys.argv[3] if len(sys.argv) >= 4 else "validation"

prefix = f"{dataset.split('/')[-1]}.{language.upper()}.{split}"

if os.path.isfile(f"data/{prefix}.seg.aligned"):
    input(f"{prefix}.seg.aligned already exists. Press enter to ignore")

dataset = load_dataset(dataset, language, split=split)

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
        id = sample['audio_id']
        path = f"/export/data2/chuber/2025/voxpopuli/{id}.mp3"
        label = sample['raw_text']
        if not label.strip():
            continue
        task((sample["audio"]["path"], path))
        seg.write(f"{id} {path}"+"\n")
        cased.write(f"{label}"+"\n")

