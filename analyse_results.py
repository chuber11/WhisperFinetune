
import subprocess
from glob import glob

def parse(lines):
    res = {}
    res["bwer"] = float(lines[2].split()[1].split("=")[1][:-1])
    res["uwer"] = float(lines[1].split()[1].split("=")[1][:-1])
    res["wer"] = float(lines[0].split()[1].split("=")[1][:-1])
    #res["conf"] = float(lines[3].split()[1])
    #res["recall"] = float(lines[3].split()[1])
    #res["precision"] = float(lines[4].split()[1])
    #res["f1"] = float(lines[5].split()[1])
    #res["pier-rest"] = float(lines[7].split()[1].split("=")[1][:-1])
    #res["pier-poi"] = float(lines[8].split()[1].split("=")[1][:-1])
    return res

def score(file, testset):
    cmd = ["bash", "score_bwer.sh",file,testset]
    r = subprocess.run(cmd, capture_output=True, text=True).stdout
    lines = [line for line in r.split("\n") if line.strip()]
    res = parse(lines)
    return res

allres = []

for testset in ["earnings","librispeech_asr.CLEAN", "librispeech_asr.OTHER","yodas"]:
    """for file in glob(f"hypos_memory/openai_whisper-large-v2.EN.data_filtered_test_{testset}_memory.EN.test.allwords.0.hyp"):
        res = score(file, testset)
        res["name"] = "baseline"
        res["testset"] = testset

        print(res)
        allres.append(res)"""

    models = [f"hypos_memory_diss/*_{testset}_memory.EN.test.allwords.*hyp"]

    for file in [f for m in models for f in glob(m)]:
        if "qwen3asr1p7B" in file:
            if "random2" in file:
                name = "qwen3asr1p7B_2"
            else:
                name = "qwen3asr1p7B_1"
        elif "qwen3omni30B" in file:
            if "random2" in file:
                name = "qwen3omni30B_2"
            else:
                name = "qwen3omni30B_1"
        elif "newwords15" in file:
            name = "newwords15"
        elif "newwords18" in file:
            name = "newwords18"
        else:
            breakpoint()

        distractors = int(file.split(".")[-3])

        if True: #try:
            res = score(file, testset)
        #except Exception as e:
        #    print(f"WARNING: Could not score {file}, {e}")
        #    continue
        res["name"] = name
        res["testset"] = testset
        res["distractors"] = distractors
        res["file"] = file

        print(res)
        allres.append(res)

import pickle
with open("analyse_results_output.pkl", "wb") as file:
    pickle.dump(allres, file)

