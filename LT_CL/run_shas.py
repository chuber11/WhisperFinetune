
import requests
import sys
from transcribe_memory import convert_mp3_to_wav_bytes
from tqdm import tqdm
from glob import glob

user = sys.argv[1]

for mp3 in tqdm(glob(f"data_to_process_{user}/*.mp3")):
    id = mp3[:-len("mp3")]

    url = "http://192.168.0.60:5010/segmenter/en+de/infer"
    audio = convert_mp3_to_wav_bytes(mp3)

    res = requests.post(url, files={"audio":audio, "max_segment_length":10, "min_segment_length":2})
    if res.status_code != 200:
        raise RuntimeError("ERROR in SHAS request")

    segmentation = res.json()

    index = 0
    with open(f"{id}seg.aligned","w") as f:
        for start, end in segmentation:
            f.write(f"{id[len(f'data_to_process_{user}/'):-1]}_{index} LT_CL/data_processed_{user}/{id[len(f'data_to_process_{user}/'):]}mp3 {start:.2f} {end:.2f}\n")
            index += 1

