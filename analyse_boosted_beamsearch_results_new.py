
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
#for testset in ["librispeech_asr.CLEAN", "librispeech_asr.OTHER"]:
    for file in glob(f"hypos_memory/openai_whisper-large-v2.EN.data_filtered_test_{testset}_memory.EN.test.allwords.0.hyp"):
        res = score(file, testset)
        res["name"] = "baseline"
        res["testset"] = testset

        print(res)
        allres.append(res)

    for file in glob(f"hypos_memory/saves_model_newwords15_checkpoint-184000.EN.data_filtered_test_{testset}_memory.EN.test.allwords.*.*.hyp"):
        if "boostall" in file:
            name = "boostall"
            distractors = int(file.split(".")[-4])
            boost = int(file.split(".")[-3])
        elif "text_filtered_other" in file:
            name = "text_real"
            distractors = int(file.split(".")[-4])
            boost = int(file.split(".")[-3])
        elif "text_filtered" in file:
            name = "text_oracle"
            distractors = int(file.split(".")[-4])
            boost = int(file.split(".")[-3])
        elif "replacements_filtered_other" in file:
            name = "replace_wrong_real"
            distractors = int(file.split(".")[-4])
            boost = int(file.split(".")[-3])
        elif "replacements_filtered" in file:
            name = "replace_wrong_oracle"
            distractors = int(file.split(".")[-4])
            boost = int(file.split(".")[-3])
        elif "replacements_other" in file:
            name = "replace_all_real"
            continue
        elif "replacement" in file:
            name = "replace_all_oracle"
            continue
        elif "inf" in file:
            continue
        else:
            name = "context_biasing"
            distractors = int(file.split(".")[-3])
            boost = int(file.split(".")[-2])

        if boost not in [0,25]:
            continue

        try:
            res = score(file, testset)
        except:
            print(f"WARNING: Could not score {file}")
            continue
        res["name"] = name
        res["testset"] = testset
        res["distractors"] = distractors
        res["boost"] = boost

        print(res)
        allres.append(res)

import pickle
with open("analyse_boosted_beamsearch_results_output_new.pkl", "wb") as file:
    pickle.dump(allres, file)

