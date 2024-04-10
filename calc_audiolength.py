
from glob import glob
from pydub import AudioSegment
from tqdm import tqdm

def get_mp3_length(file_path):
    try:
        audio = AudioSegment.from_file(file_path, format="mp3")
        length_in_seconds = len(audio) / 1000  # Convert milliseconds to seconds
        return length_in_seconds
    except Exception as e:
        print(f"Error: {e}")
        return 0


for lang in ["EN","DE"]:
    for f in glob(f"hypos/hypo_openai_whisper-large-v2_cv_filtered_beam4.{lang}.txt"):
        s = 0

        print(f)

        lines = open(f).readlines()

        for line in tqdm(lines):
            mp3 = f"/project/asr_systems/LT2022/data/{lang}/cv14.0/download/all_mp3/"+"_".join(line.split()[0].split("_")[1:])+".mp3"

            d = get_mp3_length(mp3)
            s += d

        print(f"Segfile: {f}, Hours: {s/60/60:.2f}")

