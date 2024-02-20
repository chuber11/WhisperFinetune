
import requests
import json

segfile = "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.dev.test.seg.aligned"
new_words = "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.dev.test.new_words"
references = "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.dev.test.ref"

correct = 0
total = 0

for line,line2,line3 in zip(open(segfile),open(new_words),open(references)):
    line = line.strip().split()
    new_word = line2.strip().split("|")[0]
    reference = line3.strip()

    wav = line[1]
    wav = open("/project/OML/chuber/2023/data/earnings_nw_dataset/"+wav,"rb").read()[78:]
    start = float(line[2])
    end = float(line[3])

    wav_ = wav[int(32000*start):int(32000*end)]

    use_memory = True
    if use_memory:
        memory = [w for w in new_word.split()]
        print("MEMORY:",memory)
        res = requests.post("http://192.168.0.72:5000/asr/infer/en,en", files={"pcm_s16le":wav_, "prefix": "", "memory":json.dumps(memory)})
    else:
        res = requests.post("http://192.168.0.72:5000/asr/infer/en,en", files={"pcm_s16le":wav_, "prefix": "", "memory":json.dumps([])})

    hypo = res.json()["hypo"]

    print("REF:",reference)
    print("HYPO:",hypo)
    print("NEW WORD:",new_word,new_word in hypo)

    if new_word in hypo:
        correct += 1
    total += 1

print(f"{100*correct/total:.1f}% recall")
