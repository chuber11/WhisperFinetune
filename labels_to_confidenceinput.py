
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

with open("confidence/inference/label_studio.txt", "w") as f_out:
    for f in glob("../NewsData/label_studio_source/label_studio_target/*"):
        content = json.load(open(f))
        project = content["project"]
        if project != filter_project:
            print(f"Not using label file {f} because of project filter!")
            continue
        task = content["task"]["data"]["video_url"].split("_")[-2]
        if not task in id2baselinehypos:
            breakpoint()
        mp4 = content["task"]["data"]["video_url"].split("/", maxsplit=4)[-1]

        for v in content["result"]:
            if not "text" in v["value"]:
                continue
            text = v["value"]["text"][0]
            old_text = id2baselinehypos[task][v["id"]]["text"]
            #if text == old_text:
            #    continue
            old_words = replace_except_specified_chars(old_text).split()
            new_words = [word for word in replace_except_specified_chars(text).split() if not word in old_words]

            data = {"id": "_".join([mp4,str(v["value"]["start"]),str(v["value"]["end"])]),
                    "path": "/export/data2/chuber/2024/NewsData/video/"+(mp4.replace("mp4","mp3")),
                    "start": v["value"]["start"],
                    "end": v["value"]["end"],
                    "hypo": old_text}
            res = json.dumps(data)
            f_out.write(res+"\n")
            
            #labels[v["id"]] = {"start": v["value"]["start"], "end": v["value"]["end"], "text": text, "old_text": old_text, "new_words": new_words, "mp4": mp4}
