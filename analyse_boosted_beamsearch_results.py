
import subprocess
from glob import glob

def parse(lines):
    name = "add 0  , empty m. "
    offset = 0
    if "beam" in lines[0]:
        offset += 1
    if "boostall" in lines[0] or "replacement" in lines[0]:
        offset += 1
    if not "openai" in lines[0]:
        name = lines[0].split(".")[-3-offset:-1-offset]
        distractors = int(name[0])
        add = name[1]
        name = f"add {name[1]:3s}, distr {name[0]:3s}"
    else:
        add = 0
        distractors = -1
    if "beam" in lines[0]:
        name += ", "+lines[0].split(".")[-2]+"   "
    elif "boostall" in lines[0]:
        name += ", boostall"
    elif "text_filtered_other" in lines[0]:
        name += ", text_real"
    elif "text_filtered" in lines[0]:
        name += ", text_oracle"
    elif "replacements_filtered_other" in lines[0]:
        name += ", replace_wrong_real"
    elif "replacements_filtered" in lines[0]:
        name += ", replace_wrong_oracle"
    elif "replacements_other" in lines[0]:
        name += ", replace_all_real"
    elif "replacement" in lines[0]:
        name += ", replace_all_oracle"
    else:
        name += "          "
    res = {"name": name}
    res["bwer"] = float(lines[3].split()[1].split("=")[1][:-1])
    res["uwer"] = float(lines[2].split()[1].split("=")[1][:-1])
    res["wer"] = float(lines[1].split()[1].split("=")[1][:-1])
    res["recall"] = float(lines[4].split()[1])
    res["precision"] = float(lines[5].split()[1])
    res["f1"] = float(lines[6].split()[1])
    res["pier-rest"] = float(lines[7].split()[1].split("=")[1][:-1])
    res["pier-poi"] = float(lines[8].split()[1].split("=")[1][:-1])
    res["add"] = add
    res["distractors"] = distractors
    return res

def sort(x):
    y = x.split(".")
    r = []
    if "openai" in x:
        return r

    offset = 0
    if "boostall" in x:
        r.append(-1)
        offset += 1
    elif "replacements_other" in x:
        r.append(-3)
        offset += 1
    elif "replacement" in x:
        r.append(-2)
        offset += 1

    if "beam" in y[-2]:
        r.append(-2)
        r.append(int(y[-4-offset]))
    else:
        r.append(int(y[-2-offset]))
        r.append(int(y[-3-offset]))
    return r

allres = []

files = sorted(glob("hypos_memory/*.hyp"),key=sort)
for file in files:
    #print(file)
    #if not "text" in file:
    #    continue
    #if "boostall" in file:
    #    continue
    cmd = ["bash", "score_bwer.sh",file,"earnings"]
    #print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True).stdout
    lines = [line for line in [file,*r.split("\n")] if line.strip()]
    res = parse(lines)
    print(res)
    allres.append(res)

import pickle
with open("analyse_boosted_beamsearch_results_output.pkl", "wb") as file:
    pickle.dump(allres, file)

"""res = [subprocess.run(["bash", "score_bwer.sh",file,"earnings"], capture_output=True, text=True).stdout for file in files]
lines = [line for file,r in zip(files,res) for line in [file,*r.split("\n")] if line.strip()]
res = [parse(lines[i:i+4]) for i in range(0,len(lines),4)]

for r in res:
    print(r)"""

