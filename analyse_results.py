
import subprocess
from glob import glob

def parse(lines):
    res = {}
    res["bwer"] = float(lines[2].split()[1].split("=")[1][:-1])
    res["uwer"] = float(lines[1].split()[1].split("=")[1][:-1])
    res["wer"] = float(lines[0].split()[1].split("=")[1][:-1])
    res["recall"] = float(lines[3].split()[1])
    res["precision"] = float(lines[4].split()[1])
    res["f1"] = float(lines[5].split()[1])
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

for testset in ["earnings","librispeech_asr.CLEAN", "librispeech_asr.OTHER"]:
    for file in glob(f"hypos_memory/openai_whisper-large-v2.EN.data_filtered_test_{testset}_memory.EN.test.allwords.0.hyp"):
        res = score(file, testset)
        res["name"] = "baseline"
        res["testset"] = testset

        print(res)
        allres.append(res)

    for file in glob(f"hypos_memory/saves_model_newwords18_2_checkpoint-94000*.EN.data_filtered_test_{testset}_memory.EN.test.allwords.*.*.hyp"):
        if not "baseline_adapt" in file:
            name = "context_biasing"
        else:
            name = "context_biasing_adapt"
        distractors = int(file.split(".")[-3])

        try:
            res = score(file, testset)
        except:
            print(f"WARNING: Could not score {file}")
            continue
        res["name"] = name
        res["testset"] = testset
        res["distractors"] = distractors

        print(res)
        allres.append(res)

import pickle
with open("analyse_results_output.pkl", "wb") as file:
    pickle.dump(allres, file)

