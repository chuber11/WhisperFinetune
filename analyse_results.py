
import subprocess
from glob import glob

def parse(lines):
    res = {}
    res["bwer"] = float(lines[2].split()[1].split("=")[1][:-1])
    res["uwer"] = float(lines[1].split()[1].split("=")[1][:-1])
    res["wer"] = float(lines[0].split()[1].split("=")[1][:-1])
    res["conf"] = float(lines[3].split()[1])
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

for testset in ["yodas_filtered"]: #["earnings","librispeech_asr.CLEAN", "librispeech_asr.OTHER"]:
    for file in glob(f"hypos_memory/openai_whisper-large-v2.EN.data_filtered_test_{testset}_memory.EN.test.allwords.0.hyp"):
        res = score(file, testset)
        res["name"] = "baseline"
        res["testset"] = testset

        print(res)
        allres.append(res)

    #models = [f"hypos_memory/saves_model_newwords18_*.EN.data_filtered_test_{testset}_memory.EN.test.allwords.*.*.hyp",f"hypos_memory/saves_model_qwen*.EN.data_filtered_test_{testset}_memory.EN.test.allwords.*.*.hyp"]
    models = [f"hypos_memory/saves_model_newwords18_*.EN.data_filtered_test_{testset}_memory.EN.test.allwords.*hyp"]

    for file in [f for m in models for f in glob(m)]:
        """if "qwen3asr" in file:
            name = "qwen3asr"
        elif "qwen3omni" in file:
            name = "qwen3omni"
        elif "baseline_adapt" in file and "18_2" in file:
            name = "context_biasing_18_2_adapt"
        elif "baseline_adapt" in file and "18_3" in file:
            name = "context_biasing_18_3_adapt"
        elif "18_2" in file:
            name = "context_biasing_18_2"
        elif "18_3" in file:
            name = "context_biasing_18_3"
        else:
            raise NotImplementedError(file)"""
        if "replacements_other_plus_text_logminf" in file:
            name = "replacements_other_plus_text_logminf"
        elif "replacements_other_plus_text" in file:
            name = "replacements_other_plus_text"
        elif "replacements_other_logminf" in file:
            name = "replacements_other_logminf"
            if "_new." in file:
                continue
        elif "replacements_other" in file:
            name = "replacements_other"
        elif "replacements_text_other" in file:
            name = "replacements_text_other"
        elif "oracle_text" in file:
            name = "replacements_oracle_text"
        elif "oracle" in file:
            name = "replacements_oracle"
        else:
            name = "context_biasing"

        #distractors = int(file.split(".")[-3])
        replacements = int(file.split(".")[-3])
        try:
            distractors = int(file.split(".")[-4])
        except:
            distractors = int(file.split(".")[-3])
            replacements = int(file.split(".")[-2])
            print(f"{file = }, {distractors = }, {replacements = }")

        #print(file, name)

        try:
            res = score(file, testset)
        except Exception as e:
            print(f"WARNING: Could not score {file}, {e}")
            continue
        res["name"] = name
        res["testset"] = testset
        res["distractors"] = distractors
        res["replacements"] = replacements
        res["file"] = file

        print(res)
        allres.append(res)

import pickle
with open("analyse_results_output.pkl", "wb") as file:
    pickle.dump(allres, file)

