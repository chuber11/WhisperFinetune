
import requests
import json

#wav = "data/text_1-usr0227.wav"
#segfile = "data/german_v2.stm"

wav = "/project/OML/chuber/2024/data/test.wav"
segfile = "/project/OML/chuber/2024/data/test.stm"

wav = open(wav,"rb").read()[78:]

for line in open(segfile):
    if line[:2] == ";;":
        continue
    line = line.strip().split()

    #if line[1] != "text_1-usr0227":
    #    continue

    start = float(line[4])
    end = float(line[5])
    label = " ".join(line[7:])

    wav_ = wav[int(32000*start):int(32000*end)]

    res = requests.post("http://192.168.0.72:5000/asr/infer/None,None", files={"pcm_s16le":wav_, "prefix": "", "memory":json.dumps(["Waibel"])})
    hypo = res.json()["hypo"]

    print(hypo)

    break
