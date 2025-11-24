
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm import tqdm
import os

segfiles = {}
segfiles["train"] = ["../../WhisperE+Phi2/data/cv.EN.train.seg.aligned", "../data/voxpopuli.EN.train.seg.aligned"]
segfiles["dev"] = ["../../WhisperE+Phi2/data/cv.EN.dev.seg.aligned", "../data/voxpopuli.EN.validation.seg.aligned"]

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
model = model.to("cuda")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

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

    with open(f"output/{set}.txt", "w") as f:
        for ids_,sentences_ in tqdm([[ids[i:i+length],sentences[i:i+length]] for i in range(0,len(sentences),length)]):
            for id,sentence,res in zip(ids_,sentences_,nlp(sentences_, batch_size=length)):
                had_start = False
                nes = []
                for ne in res:
                    if ne["entity"].startswith("B-"):
                        nes.append(ne["word"])
                        had_start = True
                    elif had_start and ne["entity"].startswith("I-"):
                        word = ne["word"]
                        if word.startswith("##"):
                            nes[-1] += word[2:]
                        else:
                            nes[-1] += " "+word
                    else:
                        had_start = False
                nes = [e for e in nes if e in sentence]

                if nes:
                    f.write(f"{id} {';'.join(nes)}\n")
