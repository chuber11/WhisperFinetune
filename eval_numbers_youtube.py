
from glob import glob
import json
import jiwer
import re

def get_target(line):
    ls = line.strip().split()
    category = ls[0]
    data = json.loads(" ".join(ls[1:]))
    data = [d[1] for d in data]
    return category, data

def normalize(s):
    s = re.sub(r'[^\w\s]', '', s).lower()
    return s

id2terms = {line.strip().split()[0]:get_target(line2) for line,line2 in zip(open("/export/data2/chuber/2024/YTData/test_final.seg.aligned"),open("/export/data2/chuber/2024/YTData/test_final.term"))}
id2labels = {line.strip().split()[0]:line2.strip() for line,line2 in zip(open("/export/data2/chuber/2024/YTData/test_final.seg.aligned"),open("/export/data2/chuber/2024/YTData/test_final.cased"))}

for f in sorted(glob("hypos/*youtube*")):
    print(f[len("hypos/"):])

    stats = {}
    refs = []
    hypos = []

    for line in open(f):
        ls = line.strip().split()
        id = ls[0]
        transcript = " ".join(ls[1:])

        category, terms = id2terms[id]
        label = id2labels[id]
        if not category in stats:
            stats[category] = [0,0]
        for t in terms:
            if t in transcript:
                stats[category][0] += 1
            else:
                if False: #"converted" in f: # and category == "timestamp":
                    print(f"Numeric expression = '{t}'")
                    print(f"{transcript = }")
                    print(f"{label =      }")
            stats[category][1] += 1

        refs.append(normalize(label))
        hypos.append(normalize(transcript))

    stats["avg"] = [sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())]

    for category, s in stats.items():
        print(f"    {category = :15s}: {s[0]:4d}/{s[1]:4d} = {100*s[0]/s[1]:4.1f}%")

    print("    WER: {:.1f}".format(100*jiwer.wer(refs, hypos)))

