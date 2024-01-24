
from collections import Counter
from glob import glob
import json

vocab_counter = Counter()
for f in glob("data/impairedSpeech.DE.*.cased"):
    for line in open(f):
        line = line.strip()
        for c in line:
            vocab_counter.update(c)

vocab_dict = {k: i for i, (k,v) in enumerate(vocab_counter.items()) if v>2}

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(vocab_dict)

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
