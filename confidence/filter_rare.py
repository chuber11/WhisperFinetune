
from collections import Counter
from glob import glob
from tqdm import tqdm
from data import replace_except_specified_chars
import os

lang = "EN"

files = []
for f in glob(f"../data/*{lang}*.cased"):
    files.append(f)
for f in glob(f"../data_yodas/*{lang}*.cased"):
    files.append(f)
files = [f for f in files if "train" in f or "dev" in f]

def load_trainingwords(dev=True):
    words = Counter()

    for f in files:
        if dev and not "dev" in f:
            continue
        elif not dev and "dev" in f:
            continue
        for line in tqdm(open(f).readlines()):
            line_clean = replace_except_specified_chars(line.strip())
            line_clean = line_clean.lower()
            for word in line_clean.split():
                words[word] += 1
    return words

def get_commonwords():
    dev = False

    trainingwords = load_trainingwords(dev=dev)
    trainingwords = sorted(list(trainingwords.items()),key=lambda x:-x[1])

    percent = 90
    commonwords = [w for w,c in trainingwords[:int((100-percent)/100*len(trainingwords))]]
    return commonwords

if __name__ == "__main__":
    commonwords = get_commonwords()

    remove_suffixes = ["'s","’s","'","’"]

    for f in files:
        outputfile = f"output_rare/{f.split('/')[-1]}_filtered.txt"
        if os.path.isfile(outputfile):
            input(f"File {outputfile} already exists. Continue?")
        with open(outputfile, "w") as f2:
            for line,line2 in zip(tqdm(open(f).readlines()),open(f.replace("cased","seg.aligned")).readlines()):
                line_clean = replace_except_specified_chars(line.strip())
                words_filtered = set()
                for word in line_clean.split():
                    for suffix in remove_suffixes:
                        if word.endswith(suffix):
                            word = word[:-len(suffix)]
                    word_lower = word.lower()
                    if word_lower not in commonwords:
                        words_filtered.add(word)
                if words_filtered:
                    f2.write(f"{line2.strip().split()[0]} {';'.join(words_filtered)}\n")
