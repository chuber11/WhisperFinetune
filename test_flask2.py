
import requests
import json
import time

server = "http://192.168.0.72:4999/asr/infer/None,None"
#server = "http://192.168.0.60:4999/asr/infer/None,None"
#server = "http://192.168.0.60:5008/asr/infer/None,None"
#server = "http://192.168.0.72:5008/asr/infer/None,None"

wav = "/project/OML/chuber/2024/data/test2.wav"
#wav = "/project/OML/skoneru/iwslt24/data/eval/tst2024/en-de/set3/wavs/81451.wav"
#wav = "speech-datasets/earnings21/media/4367318.wav"
#wav = "speech-datasets/earnings21/media/4361631.wav"

wav = open(wav,"rb").read()[78:]
#wav = wav[int(32000*17.9):int(32000*23.52)]
#wav = wav[int(32000*45.15):int(32000*49.09)]
#wav = wav[int(32000*23.26):int(32000*26.73)]

memory = ["AIKAASCHP","ICASP->AIKAASCHP"]
#memory = [f"AIKAASCHP"]
#memory = ["AcelRx"]
#memory = ["Llarden","Yarden->Llarden"]

print(f"{memory = }")

prefix = "Last month I was at ICASP in South Korea. Last month I was in ICASP at South Korea."
prefix = "Last month I was at"
prefix = ""

for num_beams in [1]: #,2,3,4,5]:
    t = time.time()
    res = requests.post(server, files={"pcm_s16le":wav, "prefix": prefix, "memory": json.dumps(memory), "num_beams": num_beams}) #"user": "admin@example.com"})
    hypo = res.json()
    print(f"Time: {time.time()-t:.1f}, transcript:",hypo)

