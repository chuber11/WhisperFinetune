
import re

data = {"WER EN":{},"WER DE":{},"Acc numbers human":{},"Acc numbers tts":{}}

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
            data2[d[0]] = d[1:] 
            if state2 == 5:
                set = "human" if state == 2 else "tts"
                data[f"Acc numbers {set}"][id] = float(data2["All"][-1][:-1])
        state2 = (state2+1)%6
    else:
        raise NotImplementedError

print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{|c|"+("".join(["c|" for i in range(len(data))]))+"}")
print("\\hline")
print("\\textbf{Model}"+("".join([" & \\textbf{"+k+"}" for k in data]))+" \\\\")
print("\\hline")

pattern = r"model_numbers_batchweighting(\d+)_fact(\d+)_freeze(\d+)_real_dev_data(\d+)_lr(.+)_train_emb(\d+)_checkpoint-(\d+)"

for model in data["WER EN"].keys():
    if "hypo" in model:
        model_name = model[len("hypos/hypo_"):].replace("_","/")
    else:
        match = re.match(pattern, model)
        match = [match.group(i) for i in range(1,8)]
        #model_name = f"bw {match[0]}, fact {match[1]}, freeze {match[2]}, real\_dev\_data {match[3]}, lr {match[4]}, train\_emb {match[5]}"
        model_name = f"bw {match[0]}, fact {match[1]}, freeze {match[2]}, lr {match[4]}"

    print(f"${model_name}$"+("".join(f" & {data[k][model]:.1f}\%" for k in data))+" \\\\")
    
    if "openai" in model_name:
        print("$openai/whisper-large-v2 + LLM$ & - & - & 81.1\% & 81.2\% \\\\")

print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

