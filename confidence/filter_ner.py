
from collections import Counter
from glob import glob
from tqdm import tqdm
from data import replace_except_specified_chars
import os

lower = True

def load_trainingwords(lang="EN", dev=True):
    words = Counter()
    for f in tqdm(glob(f"../data/*{lang}*.cased")):
        if dev and not "dev" in f:
            continue
        elif not dev and "dev" in f:
            continue
        for line in open(f):
            line_clean = replace_except_specified_chars(line.strip())
            if lower:
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

    for f in glob("output/*2.txt"):
        if "filtered" in f:
            continue
        print(f)
        outputfile = f[:-len(".txt")]+"_filtered.txt"
        if os.path.isfile(outputfile):
            input(f"File {outputfile} already exists. Continue?")
        with open(outputfile, "w") as f2:
            for line in tqdm(open(f).readlines()):
                parts = line.strip().split()
                id = parts[0]
                nes = parts[1].split(";")
                nes_filtered = []
                for ne in nes:
                    ne_clean = replace_except_specified_chars(ne)
                    if lower:
                        ne_clean = ne_clean.lower()
                    if any(w not in commonwords for w in ne_clean.split()):
                        nes_filtered.append(ne)
                        break
                if nes_filtered:
                    f2.write(f"{id} {';'.join(nes_filtered)}\n")
