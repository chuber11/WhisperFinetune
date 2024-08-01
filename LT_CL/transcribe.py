
from glob import glob
from tqdm import tqdm
import requests
import os
import sys
import time

user = sys.argv[1]
audio_path = f"data_to_process_{user}"

while True:
    try:
        res = requests.post("http://192.168.0.60:5008/inference")
    except requests.exceptions.ConnectionError:
        print("Model not running, waiting...")
        time.sleep(10)
    else:
        break

for mp3 in tqdm(glob(f"{audio_path}/*.mp3")):
    hypofile = mp3[:-len("mp3")] + "hypo.json"

    if os.path.isfile(hypofile):
        continue

    mp3 = open(mp3,"rb").read()
    res = requests.post("http://192.168.0.60:5008/inference", files={"audio":mp3}) # with running ASR model from https://gitlab.kit.edu/kit/isl-ai4lt/lt-middleware/inference/faster-whisper.git

    with open(hypofile, "w") as f:
        f.write(res.text)

