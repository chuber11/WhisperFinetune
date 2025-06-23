
from glob import glob

def calc(segfile):
    s = 0
    anz = 0
    for line in open(segfile):
        _, _, start, end = line.strip().split()
        s += float(end) - float(start)
        anz += 1
    print(f"Segfile: {segfile}, Hours: {s/60/60:.1f}, Anz: {anz}")

for segfile in glob("data_impairedSpeech*/*.seg.aligned"):
    if "correction" in segfile:
        continue
    calc(segfile)
