
from glob import glob
from collections import Counter, defaultdict
import re
from tqdm import tqdm

def replace_except_specified_chars(text):
    # This pattern matches any character that is NOT a-z, A-Z, äöüÄÖÜß
    pattern = r'[^a-zA-ZäöüÄÖÜß\'–—’&]'
    # Replace these matched characters with a space
    result = re.sub(r'\s+', ' ', re.sub(pattern, ' ', text))
    return result

set_ = "*"

data = []
for f in glob(f"data_yodas/EN.{set_}.cased"):
    f2 = f.replace("cased","seg.aligned")
    for line,line2 in zip(open(f2),tqdm(open(f).readlines())):
        seg_id = line.strip().split()[0]
        id = "_".join(seg_id.split("_")[:-4])
        line_clean = replace_except_specified_chars(line2.strip())
        words = set(line_clean.split())
        data.append((id, words, line, line2))

id2words = defaultdict(Counter)
for id, words, line, line2 in data:
    for word in words:
        id2words[id][word] += 1

words = Counter()
for id,counter in id2words.items():
    for word,c in counter.items():
        words[word] += c

filter_start = ["'","’","'"]
filter_end = ["'s","’s","s'","s’"]
filter_end2 = ["'","’","s"]

id2words2 = defaultdict(Counter)
for id,counter in id2words.items():
    counter2 = Counter()
    for word,c in counter.items():
        if word[0] in filter_start or word[-2:] in filter_end or word[-1] in filter_end2:
            continue
        if words[word]+words[word.lower()] > c:
            continue
        counter2[word] = c
    id2words2[id] = counter2
id2words = id2words2

while True:
    threshold = int(input(f"Current threshold: ")) #4
    threshold2 = int(input(f"Current threshold2: ")) #8

    word2id = {}
    for id,counter in id2words.items():
        for word,c in counter.items():
            if c < threshold or c >= threshold2:
                continue
            word2id[word] = id
    print(f"Threshold: {threshold}, threshold2: {threshold2}, rare words: {len(word2id)}")

    confirm = input("Is this okay? (y/n): ")
    if confirm.lower() != "y":
        continue

    with open(f"data_filtered_test/yodas_memory.EN.test.seg.aligned","w") as g, \
         open(f"data_filtered_test/yodas_memory.EN.test.cased","w") as g2, \
         open(f"data_filtered_test/yodas_memory.EN.test.words","w") as g3:
        for id, words_, line, line2 in data:
            found_words = set(word for word in words_ if word in word2id and word2id[word] == id)

            if found_words:
                g.write(line)
                g2.write(line2)
                g3.write("|".join(found_words)+"\n")
                #print(found_words)
