
from data import replace_except_specified_chars
from glob import glob
from collections import Counter
from tqdm import tqdm

def load_trainingwords(lang="EN", dev=True, lower=True):
    words = Counter()
    for f in tqdm(glob(f"data/*{lang}*.cased")):
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

#datasets = ["voxpopuli.EN", "fleurs.EN_US", "librispeech_asr.CLEAN", "librispeech_asr.OTHER"]
datasets = ["librispeech_asr.CLEAN", "librispeech_asr.OTHER"]
dev = False
lower = True

trainingwords = load_trainingwords(dev=dev, lower=lower)
trainingwords = sorted(list(trainingwords.items()),key=lambda x:-x[1])

percent = 90
commonwords = [w for w,c in trainingwords[:int((100-percent)/100*len(trainingwords))]]

for dataset in datasets:
    with open(f"data_filtered_test/{dataset}.test.words","w") as f:
        lines = set()
        for line in open(f"data_filtered_test/{dataset}.test.cased"):
            if line not in lines:
                lines.add(line)
            else:
                f.write("\n")
                continue
            rarewords = set()
            for word in replace_except_specified_chars(line.strip()).split():
                word_ = word if not lower else word.lower()
                if word_ not in commonwords:
                    rarewords.add(word)
            f.write("|".join(rarewords)+"\n")

for dataset in datasets:
    segfile = f"data_filtered_test/{dataset}.test.seg.aligned"
    labelfile = f"data_filtered_test/{dataset}.test.cased"
    wordfile = f"data_filtered_test/{dataset}.test.words"

    words = Counter()
    for words_ in open(wordfile):
        for word in words_.strip().split("|"):
            words[word] += 1

    segfile_out = f"data_filtered_test/{dataset}_memory.EN.test.seg.aligned"
    labelfile_out = f"data_filtered_test/{dataset}_memory.EN.test.cased"
    wordfile_out = f"data_filtered_test/{dataset}_memory.EN.test.words"

    lines = set()
    allwords = set()

    with open(segfile_out,"w") as f, open(labelfile_out,"w") as f2, open(wordfile_out,"w") as f3:
        for line, line2, line3 in zip(open(segfile),open(labelfile),open(wordfile)):
            if not line in lines:
                lines.add(line)
            else:
                continue
            words_ = line3.strip()
            if not words_:
                continue
            words_ = [word for word in words_.split("|") if words[word] > 1]
            if not words_:
                continue
            for word in words_:
                allwords.add(word)
            line3 = "|".join(words_)+"\n"
            f.write(line)
            f2.write(line2)
            f3.write(line3)

    with open(f"data_filtered_test/{dataset}_memory.EN.test.allwords","w") as f:
        for word in allwords:
            f.write(f"{word}\n")

