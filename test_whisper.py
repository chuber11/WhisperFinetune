
import requests
import json
import threading
import time

def req():
    server = "http://192.168.0.72:5008/asr/infer/None,None"

    wav = "/project/OML/chuber/2024/data/test2.wav"
    wav = open(wav,"rb").read()[78:]

    res = requests.post(server, files={"pcm_s16le":wav})
    hypo = res.json()["hypo"]

    print(hypo)

def req_in_thread():
    t = threading.Thread(target=req)
    t.start()
    return t

t = time.time()
ths = []
for i in range(20):
    th = req_in_thread()
    ths.append(th)

for th in ths:
    th.join()
print(time.time()-t)
