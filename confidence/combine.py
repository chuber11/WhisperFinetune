
import json
import random
from tqdm import tqdm
from glob import glob

random.seed(42)

segfiles = []
for f in glob(f"../data/*EN*.seg.aligned"):
    segfiles.append(f)
for f in glob(f"../data_yodas/*EN*.seg.aligned"):
    segfiles.append(f)
segfiles = [f for f in segfiles if "train" in f or "dev" in f]

segfiles2 = {}
segfiles2["train"] = [f for f in segfiles if "train" in f and ("cv" in f or "yodas" in f)]
segfiles2["dev"] = [f for f in segfiles if "dev" in f and ("cv" in f or "yodas" in f)]
segfiles = segfiles2

for set_ in ["dev","train"]:
    print(f"Combining set {set_}")

    with open(f"output_combined/{set_}.txt", "w") as f:
        path2data = {}
        for file in tqdm(glob(f"output_conf/*{set_}*.txt")):
            for line in open(file):
                data = json.loads(line)
                path = data["path"]
                if not path in path2data:
                    path2data[path] = []
                path2data[path].append(data)

        ids = set()
        n = 0
        for data in tqdm(path2data.values()):
            data_ = {"id": data[0]["id"], "path": data[0]["path"], "label": data[0]["label"], "nes":[]}
            ids.add(data_["id"])
            for v in data:
                prefix = v["prefix"]
                suffix = v["suffix"]
                ne = v["ne"]
                ne_hypo = v["hypo_to_score"][0][0][len(prefix):-len(suffix)].strip()
                data_["nes"].append({"ne_hypo":ne_hypo, "ne": ne})
            f.write(json.dumps(data_) + "\n")
            n += 1
            
        # add examples without NEs
        data = []
        for segfile in tqdm(segfiles[set_]):
            for line,line2 in zip(open(segfile),open(segfile.replace(".seg.aligned",".cased"))):
                id, path = line.strip().split()
                label = line2.strip()
                if id not in ids:
                    data.append({"id": id, "path": path, "label": label, "nes":[]})

        for _ in tqdm(range(n)):
            idx = random.randint(0, len(data)-1)
            d = data.pop(idx)
            f.write(json.dumps(d) + "\n")
