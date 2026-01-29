
import subprocess
from tqdm import tqdm

with open("/home/chuber/2025/data/earnings21/earnings_memory.EN.test.seg.aligned", "w") as f:
    lines = open("data_filtered_test/earnings_memory.EN.test.seg.aligned").readlines()
    for line in tqdm(lines):
        id, mp3, start, end = line.strip().split()
        id2 = f"{id}_{start}_{end}"

        # use subprocess ffmpeg to extract audio from start to end and save it under /home/chuber/2025/data/earnings21/mp3/
        output_path = f"/home/chuber/2025/data/earnings21/mp3/{id2}.mp3"

        cmd = ["ffmpeg", "-y", "-i", mp3, "-ss", start, "-to", end, output_path]

        subprocess.run(cmd)
        f.write(f"{id} {output_path}\n")
