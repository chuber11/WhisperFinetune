
import requests
import json

split = "test"

segfile = "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw."+split+".test.seg.aligned"
new_words = "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw."+split+".test.new_words"
references = "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw."+split+".test.ref"

correct_nomem = 0
correct_mem = 0
total = 0

for line,line2,line3 in zip(open(segfile),open(new_words),open(references)):
    line = line.strip().split()
    new_word = line2.strip().split("|")[0]
    reference = line3.strip()

    wav = line[1]
    wav = open("/project/OML/chuber/2023/data/earnings_nw_dataset/"+wav,"rb").read()[78:]
    start = float(line[2])
    end = float(line[3])

    wav_ = wav[int(16000*start)*2:int(16000*end)*2]

    memory = [w for w in new_word.split()]
    print("MEMORY:",memory)

    res = requests.post("http://192.168.0.72:5000/asr/infer/en,en", files={"pcm_s16le":wav_, "prefix": "", "memory":json.dumps(memory)})
    hypo_mem = res.json()["hypo"]
    
    res = requests.post("http://192.168.0.72:5000/asr/infer/en,en", files={"pcm_s16le":wav_, "prefix": "", "memory":json.dumps([])})
    hypo_nomem = res.json()["hypo"]

    print("REF:      ",reference)
    print("HYPOMEM:  ",hypo_mem)
    print("HYPONOMEM:",hypo_nomem)
    print("NEW WORD:",new_word,new_word in hypo_mem)

    if new_word in hypo_nomem:
        correct_nomem += 1
    if new_word in hypo_mem:
        correct_mem += 1
    total += 1

print(f"No mem: {100*correct_nomem/total:.1f}% recall")
print(f"Mem:    {100*correct_mem/total:.1f}% recall")
