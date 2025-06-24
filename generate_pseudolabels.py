
import sys
import json
from glob import glob
from collections import Counter
import re

def replace_except_specified_chars(text):
    # This pattern matches any character that is NOT a-z, A-Z, äöüÄÖÜß
    pattern = r'[^a-zA-ZäöüÄÖÜß]'
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
    if j>i:
        break

    for line in open(f"{datadir}/memory_files/{memoryfilesname}/{talk_}.memory"):
        new_words.add(line.strip())

# Count the number new words already occured to later split between train and dev sets
counters = {"train":Counter(), "dev":Counter()}
for split in ["train","dev"]:
    counter = counters[split]
    for file in glob(f"CL/{experimentname}/data/*.{split}.new_words"):
        for line in open(file):
            new_words = line.strip().split("|")
            for w in new_words:
                counter[w] += 1

# Open files for new pseudolabels to write them out
outfiles = {}
for split in ["train","dev"]:
    for type in ["seg.aligned","ref","new_words"]:
        outfiles[(split,type)] = open(f"CL/{experimentname}/data/{talk}.{split}.{type}","w")

# Write out new pseudolabels
for line,line2,line3 in zip(open(f"{datadir}/segfiles/{talk}.seg.aligned"),open(f"CL/{experimentname}/hypos/{talk}.hyp"),open(f"{datadir}/segfiles/{talk}.hypo")):
    seg = line.strip().split()
    hypo = line2.strip().split()
    hypo_baseline = line3.strip().split()
    id, hypo = hypo[0], " ".join(hypo[1:])

    if seg[0] != id:
        print("ERROR: segfile and hypofile not aligned!")
        continue

    found_new_words = [new_word for new_word in new_words if new_word in replace_except_specified_chars(hypo).split() and new_word not in replace_except_specified_chars(hypo_baseline).split()]
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

print(sum(counters["train"].values()))

