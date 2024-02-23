
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

# usage: python generate_pseudolabels.py $talk $i $datadir $experimentdir

talk = sys.argv[1]
i = int(sys.argv[2])
datadir = sys.argv[3]
experimentdir = sys.argv[4]

# Load new_words that have been written in the memory until now to later extract pseudolabels containing them
new_words = set()
for j,talk_ in enumerate(open(datadir+"/order.txt")):
    if j>i:
        break

    talk_ = json.loads(talk_.strip())
    id = talk_[0][len("audio/"):-len(".info.json")]

    for line in open(datadir+"/memory_files/"+id+".memory"):
        new_words.add(line.strip())

# Count the number new words already occured to later split between train and dev sets
new_word_to_number_occ_train = Counter()
new_word_to_number_occ_dev = Counter()
for split, counter in zip(["train","dev"],[new_word_to_number_occ_train,new_word_to_number_occ_dev]):
    for file in glob(f"{experimentdir}/data/*.{split}.new_words"):
        for line in open(file):
            new_words = line.strip().split("|")
            for w in new_words:
                counter[w] += 1

# Open files for new pseudolabels to write them out
outfiles = {}
for split in ["train","dev"]:
    for type in ["seg.aligned","ref","new_words"]:
        outfiles[(split,type)] = open(f"{experimentdir}/data/{talk}.{split}.{type}","w")

# Write out new pseudolabels
for line,line2 in zip(open(f"{datadir}/segfiles/{talk}.seg.aligned"),open(f"{experimentdir}/hypos/{talk}.hyp")):
    seg = line.strip().split()
    hypo = line2.strip().split()
    id, hypo = hypo[0], " ".join(hypo[1:])

    if seg[0] != id:
        print("ERROR: segfile and hypofile not aligned!")
        continue

    found_new_words = [new_word for new_word in new_words if new_word in replace_except_specified_chars(hypo).split()]
    if not found_new_words:
        continue

    split = "train"
    if all(w in new_word_to_number_occ_train for w in found_new_words) and not all(w in new_word_to_number_occ_dev for w in found_new_words):
        split = "dev"

    outfiles[(split,"seg.aligned")].write(line)
    outfiles[(split,"ref")].write(hypo+"\n")
    outfiles[(split,"new_words")].write("|".join(found_new_words)+"\n")

