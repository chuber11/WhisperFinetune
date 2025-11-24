
import json
import sys

i = int(sys.argv[1])

data = [json.loads(line) for line in open("output_conf/dev.txt")]

s = 0
for item in data:
    ne = item["ne"]
    hypo = item["hypo"][i]
    if ne in hypo:
        s += 1 
print(s, len(data), s/len(data))
