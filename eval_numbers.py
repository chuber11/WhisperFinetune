
import sys
from glob import glob
from NumbersTrainingset.split import extract_numbers

hypofiles = sys.argv[1] #"hypos/hypo_saves_model_numbers16_checkpoint-2000_beam4.*.txt"
split = sys.argv[2] #"test"
segfiles = sys.argv[3] if len(sys.argv)>=4 else f"data_numbers/llm_augment.*.{split}.seg.aligned"

id_to_label = {}
for segfile in glob(segfiles):
    for line,line2,line3 in zip(open(segfile),open(segfile.replace("seg.aligned","cased")),open(segfile.replace("seg.aligned","type"))):
        id = line.strip().split()[0]
        label = line2.strip()
        type = line3.strip()
        id_to_label[id] = (label,type)

correct = {}
total = {}
for hypofile in glob(hypofiles):
    for line in open(hypofile):
        line = line.strip().split()
        id, hypo = line[0], " ".join(line[1:])

        if not id in id_to_label:
            continue
        label,type = id_to_label[id]
        numbers = extract_numbers(label)

        for n in numbers:
            if not type in correct:
                if n in hypo:
                    correct[type] = 1
                else:
                    correct[type] = 0
                total[type] = 1
            else:
                if n in hypo:
                    correct[type] += 1
                else: #if type == "currency":
                    pass #print(hypo,n)
                total[type] += 1

for (type,c),(_,t) in zip(correct.items(),total.items()):
    print(f"{type = :9s}, {c = :3d}, {t = :3d}, {100*c/t = :4.1f}%")

c = sum(correct.values())
t = sum(total.values())
print(f"All             , {c = :3d}, {t = :3d}, {100*c/t = :4.1f}%")

