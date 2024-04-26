
from glob import glob
import pdftotext
import re
import io

def pdf_to_text(file_bytes):
    for page in pdftotext.PDF(io.BytesIO(file_bytes)):
        yield page

def text_to_words(text):
    words = set()
    for line in text.split("\n"):
        line = clean_line(line.strip())
        #print(line)
        for word in line.split():
            words.add(word)
    return words

remove = [",",";",".",":","!","?","•","(",")","{","}","…",'"',"[","]","“","”","✔","●"]

def clean_line(line):
    for r in remove:
        line = line.replace(r," ")
    line = line.replace("’","'")
    while True:
        line2 = line.replace("  "," ")
        if len(line2)==len(line):
            break
        line = line2
    return line

training_labels = None
def load_traininglabels():
    global training_labels
    if training_labels is None:
        print("Loading training labels...")
        training_labels = set(w.lower() for lang in ["EN","DE"] for line in open(f"data/cv.{lang}.train.cased", encoding="utf-8") for w in line.strip().split())
    return training_labels

def extract_words(file_bytes):
    training_labels = load_traininglabels()

    text_pdf = pdf_to_text(file_bytes) # extract text of pdf

    words = set()
    lower_to_word = {}
    for i,t in enumerate(text_pdf):
        words_i = text_to_words(t)
        for w in words_i:
            if w.lower() not in training_labels and not any(n in w for n in "0123456789") and re.fullmatch(r"[a-zA-Z]+[a-zA-Z\-]?[a-zA-Z]+",w):
                if w.lower() in lower_to_word:
                    w = lower_to_word[w.lower()]
                words.add(w)
                lower_to_word[w.lower()] = w

    return sorted(list(words))

