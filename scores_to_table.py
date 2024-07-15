
import re

data = {"WER EN":{},"WER DE":{},"WER Numbers EN":{},"WER Numbers DE":{}}

state = -1
state2 = 0
data2 = {}
for line in open("all_number_scores.txt"):
    line = line.strip()
    if line[:4] == "----":
        state += 1
        continue

    if state in [0,1]:
        if state2 == 0:
            id = line
        elif state2 == 3:
            wer = line.split()[1]
            lang = "EN" if state == 0 else "DE"
            data[f"WER {lang}"][id] = 100*float(wer)
        state2 = (state2+1)%5
    elif state in [2,3]:
        if state2 == 0:
            id = line
        elif state2 in [1,2,3,4,5]:
            d = [x.split("=")[-1].strip() for x in line.split(",")]
            data2[d[0]] = [float(d[1]),float(d[2]),float(d[3][:-1])]
            if state2 == 5:
                set = "human" if state == 2 else "tts"
                for k,v in data2.items():
                    k2 = f"Acc numbers {set} {k}"
                    if k2 not in data:
                        data[k2] = {}
                    data[k2][id] = v
                data2 = {}
        state2 = (state2+1)%6
    elif state == 4:
        if state2 == 0:
            id = line
        elif state2 in [1,2,3]:
            d = [x.split("=")[-1].strip() for x in line.split(",")]
            data2[d[0]] = [float(d[1]),float(d[2]),float(d[3][:-1])]
            if state2 == 3:
                for k,v in data2.items():
                    k2 = f"Acc numbers human train {k}"
                    if k2 not in data:
                        data[k2] = {}
                    data[k2][id] = v
                data2 = {}
        state2 = (state2+1)%4
    elif state in [5,6]:
        if state2 == 0:
            id = line
            if id == "openai_whisper-large-v2_converted":
                id = "openai_whisper-large-v2 + gpt4-turbo"
            """elif id == "openai_whisper-large-v2+textseg1_beam4.wer":
                id = "openai_whisper-large-v2 + textseg1_beam4.wer"
            elif id == "openai_whisper-large-v2+textseg2_beam4.wer":
                id = "openai_whisper-large-v2 + textseg2_beam4.wer"""
            if id.startswith("saves"):
                id = id[len("saves_"):]
        elif state2 == 3:
            wer = line.strip().split(": ")[1]
            lang = "EN" if state==5 else "DE"
            data[f"WER Numbers {lang}"][id] = 100*float(wer)
        state2 = (state2+1)%5
    else:
        raise NotImplementedError

def map(k):
    replaces = [("numbers human ",""),
                ("year","years"),
                ("timestamp","timestamps"),
                ("currency","currency amounts"),
                ("number","quantities"),
                ("train ",""),
                ("Acc","Acc. (\%)"),
                ("WER","WER (\%)"),
            ]
    for r1,r2 in replaces:
        k = k.replace(r1,r2)
    return k

columns = [k for k in data.keys() if not "tts" in k and "All" not in k and not "Numbers" in k]
columns.insert(2, columns.pop(3))
columns.insert(6, columns.pop(-1))
columns.insert(2,"WER Numbers EN")
columns.insert(3,"WER Numbers DE")

allcolumns = [columns[:4],columns[4:8]]

for columns in allcolumns:
    print("\\begin{table*}[h]")
    print("\\centering")
    print("\\begin{tabular}{|T{0.23\\textwidth}||"+("".join(["T{0.065\\textwidth}|" for i in range(len(columns))]))+"}")
    print("\\hline")
    print("Model"+("".join([" & "+map(k) for k in columns]))+" \\\\")
    print("\\hline")

    pattern = r"model_numbers_batchweighting(\d+)_fact(\d+)_freeze(\d+)_real_dev_data(\d+)_lr(.+)_train_emb(\d+)_checkpoint-(\d+)"
    pattern2 = r"model_numbers_baseline_fact(\d+)_freeze(\d+)_real_dev_data(\d+)_lr(.+)_train_emb(\d+)_checkpoint-(\d+)"

    for model in data["WER EN"].keys():
        if "openai" in model:
            model_name = model.replace("_","/")
            model_name = model_name.replace("openai/whisper-large-v2","baseline")
        elif "baseline" in model:
            #continue
            match = re.match(pattern2, model)
            match = [match.group(i) for i in range(1,7)]
            #model_name = f"baseline, fact {match[0]}, freeze {match[1]}, real\_dev\_data {match[2]}, lr {match[3]}, train\_emb {match[4]}"
            model_name = f"fine-tuning old data, fact {match[0]}, freeze {match[1]}, lr {match[3]}, train\_emb {match[4]}"
        else:
            match = re.match(pattern, model)
            match = [match.group(i) for i in range(1,8)]
            #model_name = f"bw {match[0]}, fact {match[1]}, freeze {match[2]}, real\_dev\_data {match[3]}, lr {match[4]}, train\_emb {match[5]}"
            #if match[0]!="0" or match[1]!="0" or match[2]!="0" or match[4]!="1e-6":
            #    continue
            model_name = f"bw {match[0]}, fact {match[1]}, freeze {match[2]}, lr {match[4]}, train\_emb {match[5]}"
            #model_name = "fine-tuning"

        s = f"{model_name}"
        for k in columns:
            if 'WER' in k:
                if model not in data[k]:
                    print([k for k in data[k].keys() if "350" in k])
                    print("WARNING",model)
                    sys.exit()
                    continue
                s += f" & {data[k][model]:.1f}"
            elif 'train' in k:
                s += f" & {data[k][model][-1]:.1f}" if model in data[k] else " & "
            else:
                s += f" & {data[k][model][-1]:.1f}"
        s += " \\\\"
        print(s)
        
    print("\\hline")
    print("\\end{tabular}")
    #print("\\end{table}")

