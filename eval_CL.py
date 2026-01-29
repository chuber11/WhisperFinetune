
from glob import glob
import json
import torch
import re

filter_project = 15

def replace_except_specified_chars(text):
    # This pattern matches any character that is NOT a-z, A-Z, äöüÄÖÜß
    pattern = r'[^a-zA-ZäöüÄÖÜß]' #’]'
    # Replace these matched characters with a space
    result = re.sub(r'\s+', ' ', re.sub(pattern, ' ', text))
    return result

# Load baseline hypos
id2baselinehypos = {}
for f in glob("../NewsData/label_studio_source/*/*.labelstudio.json"):
    id = f.split("/")[-2]
    content = json.load(open(f))
    hypos = {}
    for v in content["predictions"][0]["result"]:
        if not "text" in v["value"]:
            continue
        v["value"]["text"] = v["value"]["text"][0]
        del v["value"]["channel"]
        hypos[v["id"]] = v["value"]

    id2baselinehypos[id] = hypos
#print(id2baselinehypos)

# Load labels
id2changedlabels = {}
for f in glob("../NewsData/label_studio_source/label_studio_target/*"):
    content = json.load(open(f))
    project = content["project"]
    if project != filter_project:
        print(f"Not using label file {f} because of project filter!")
        continue
    task = content["task"]["data"]["video_url"].split("_")[-2]
    if not task in id2baselinehypos:
        breakpoint()
    labels = {}
    for v in content["result"]:
        if not "text" in v["value"]:
            continue
        text = v["value"]["text"][0]
        old_text = id2baselinehypos[task][v["id"]]["text"]
        if text == old_text:
            continue
        old_words = replace_except_specified_chars(old_text).split()
        new_words = [word for word in replace_except_specified_chars(text).split() if not word in old_words]
        labels[v["id"]] = {"start": v["value"]["start"], "end": v["value"]["end"], "text": text, "old_text": old_text, "new_words": new_words}

    if labels:
        id2changedlabels[id] = labels

"""for id, labels in id2changedlabels.items():
    print(f"ID: {id}")
    for seg_id, data in labels.items():
        print(f" Segment ID: {seg_id}")
        print(f"  Old Text: {data['old_text']}")
        print(f"  New Text: {data['text']}")
        print(f"  New Words: {data['new_words']}")"""

articles_new_words = torch.load("../NewsData/articles_words_rare_in_training.pt")
#print(articles_new_words['https://www.nbcnews.com/select/shopping/best-scalp-massagers-rcna198370'])
#breakpoint()

id2newwordsinarticles = {}
for id, labels in id2changedlabels.items():
    newwordsinarticles = {}
    for seg_id, data in labels.items():
        for new_word in data["new_words"]:
            new_word = "Emilie Ikeda"
            article_ids = set(article_id for article_id, counter in articles_new_words.items() if new_word in counter)
            if article_ids:
                pass
            breakpoint()
    if newwordsinarticles:
        id2newwordsinarticles[id] = newwordsinarticles

