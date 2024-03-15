
import sys
import re

def clean(text):
    pattern = r'[^a-zA-ZäöüÄÖÜß]'
    # Replace these matched characters with a space
    result = re.sub(r'\s+', ' ', re.sub(pattern, ' ', text))
    return result

infile = sys.argv[1] #"hypos/hypo_saves_model5_checkpoint-180_beam4.txt"
outfile = sys.argv[2] #"hypos/hypo_saves_model5_checkpoint-180_beam4_ipa.txt"
hypos = sys.argv[3] == "True"

word_to_ipa = {}
for line in open("de_word_ipa.csv"):
    line = line.strip().split(",")
    word, ipa = line[0], line[1]
    word_to_ipa[word.lower()] = ipa

with open(outfile, "w") as f:
    for line in open(infile):
        if hypos:
            line = line.strip().split()
            id, hypo = line[0], " ".join(line[1:])
        else:
            line = line.strip()
            id, hypo = line.split("\t")

        words = [w.lower() for w in clean(hypo).split()]
        characters = " ".join(c for w in words for c in word_to_ipa.get(w, "|"))

        if hypos:
            out = f"{id} {characters}\n"
        else:
            out = f"{id}\t{characters}\n"

        f.write(out)

