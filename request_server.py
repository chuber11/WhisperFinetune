
import requests
import sys

server = sys.argv[1]
wav = open(sys.argv[2],"rb").read()[78:]
prefix = sys.argv[3]

res = requests.post(server, files={"pcm_s16le":wav, "prev_text_tokens": prefix})

hypo = res.json()["hypo"]
print(hypo)

