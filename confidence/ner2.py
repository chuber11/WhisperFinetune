

from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
import os

segfiles = {}
segfiles["train"] = ["../../WhisperE+Phi2/data/cv.EN.train.seg.aligned", "../data/voxpopuli.EN.train.seg.aligned"]
segfiles["dev"] = ["../../WhisperE+Phi2/data/cv.EN.dev.seg.aligned", "../data/voxpopuli.EN.validation.seg.aligned"]

# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")
tagger.to('cuda')

#sentences = ["My name is Wolfgang and I live in Berlin.", "My name is Alexander Waibel and I'm professor in Karlsruhe.","George Washington went to Washington."]

for set in ["dev","train"]:
    ids = []
    sentences = []
    for segfile in segfiles[set]:
        for line,line2 in zip(open(segfile),open(segfile.replace("seg.aligned","cased"))):
            id = line.split()[0]
            sentence = line2.strip()
            ids.append(id)
            sentences.append(sentence)

    length = 64

    if os.path.isfile(f"output/{set}.txt"):
        continue

    with open(f"output/{set}2.txt", "w") as f:
        for ids_,sentences_ in tqdm([[ids[i:i+length],sentences[i:i+length]] for i in range(0,len(sentences),length)]):
            sentences_ = [Sentence(s) for s in sentences_]
            tagger.predict(sentences_, mini_batch_size=length)

            for id,sentence in zip(ids_,sentences_):
                nes = [r.text for r in sentence.get_spans('ner')]
                if nes:
                    f.write(f"{id} {';'.join(nes)}\n")
