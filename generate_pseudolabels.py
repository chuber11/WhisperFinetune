
import sys
import json
from glob import glob
from collections import Counter
import re
import os

def replace_except_specified_chars(text):
    # This pattern matches any character that is NOT a-z, A-Z, äöüÄÖÜß
    pattern = r'[^a-zA-ZäöüÄÖÜß\'–—’&]'
    # Replace these matched characters with a space
    result = re.sub(r'\s+', ' ', re.sub(pattern, ' ', text))
    return result

# usage: python generate_pseudolabels.py $talk $i $datadir $experimentname $memoryfilesname

talk = sys.argv[1]
i = int(sys.argv[2])
datadir = sys.argv[3]
experimentname = sys.argv[4]
memoryfilesname = sys.argv[5]

# Load new_words that have been written in the memory until now to later extract pseudolabels containing them
new_words = set()
for j,talk_ in enumerate(open(f"{datadir}/memory_files/order_{memoryfilesname}.txt")):
    talk_ = talk_.strip()
    if j>i:
        break

    for line in open(f"{datadir}/memory_files/{memoryfilesname}/{talk_}.memory"):
        new_words.add(line.strip())

# Count the number new words already occured to later split between train and dev sets
counters = {"train":Counter(), "dev":Counter()}
for j,talk_ in enumerate(open(f"{datadir}/memory_files/order_{memoryfilesname}.txt")):
    talk_ = talk_.strip()
    if j>=i:
        break

    for split in ["train","dev"]:
        counter = counters[split]
        for line in open(f"CL/{experimentname}/data/{talk_}.{split}.new_words"):
            new_words_ = line.strip().split("|")
            for w in new_words_:
                counter[w] += 1

# Open files for new pseudolabels to write them out
outfiles = {}
for split in ["train","dev"]:
    for type in ["seg.aligned","ref","new_words"]:
        outfile = f"CL/{experimentname}/data/{talk}.{split}.{type}"
        """if os.path.isfile(outfile):
            n_found = sum(counters["train"].values())
            print(n_found)
            sys.exit()"""
        outfiles[(split,type)] = open(outfile,"w")

# Write out new pseudolabels
#for line,line2,line3,line4 in zip(open(f"{datadir}/segfiles/{talk}.seg.aligned"),open(f"CL/{experimentname}/hypos/{talk}.hyp"),open(f"{datadir}/segfiles/hypos_baseline_adapt_yodas/{talk}.hypo"),open(f"{datadir}/segfiles/hypos_baseline_qwen/{talk}.hyp")):
for line,line2,line3,line4 in zip(open(f"{datadir}/segfiles/{talk}.seg.aligned"),open(f"CL/{experimentname}/hypos/{talk}.hyp"),open(f"{datadir}/segfiles/hypos_baseline_before_adapt/{talk}.hypo"),open(f"{datadir}/segfiles/hypos_baseline_qwen/{talk}.hyp")):
    seg = line.strip().split()

    line2 = line2.strip().split()
    id = line2[0]
    hypo = " ".join(line2[1:])

    line3 = line3.strip().split()
    id2 = line3[0]
    hypo_baseline = " ".join(line3[1:])

    line4 = line4.strip().split()
    id3 = line4[0]
    hypo_baseline2 = " ".join(line4[1:])

    if not (seg[0] == id == id2 == id3):
        print("ERROR: segfile and hypofile not aligned!")
        continue

    hypo_ = replace_except_specified_chars(hypo).lower().split()
    hypo_baseline = replace_except_specified_chars(hypo_baseline).lower().split()
    hypo_baseline2 = replace_except_specified_chars(hypo_baseline2).lower().split()

    #found_new_words = [new_word for new_word in new_words if new_word in hypo_]
    found_new_words = [new_word for new_word in new_words if new_word.lower() in hypo_ and new_word.lower() not in hypo_baseline and new_word.lower() not in hypo_baseline2]
    if not found_new_words:
        continue

    split = "train"
    if all(w in counters["train"] for w in found_new_words) and not all(w in counters["dev"] for w in found_new_words):
        split = "dev"

    outfiles[(split,"seg.aligned")].write(line)
    outfiles[(split,"ref")].write(hypo+"\n")
    outfiles[(split,"new_words")].write("|".join(found_new_words)+"\n")

    for w in found_new_words:
        counters[split][w] += 1

n_found = sum(counters["train"].values())
print(n_found)
if n_found > 0:
    import time
    time.sleep(1)

