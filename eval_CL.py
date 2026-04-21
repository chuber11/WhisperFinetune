
from glob import glob

#approach1_context_path = "/export/data2/chuber/2024/Whisper+Memory/CL/cb_only_memoryfiles1/data"
#approach2_context_path = "/export/data2/chuber/2024/Whisper+Memory/CL/qwen0/hypos"
approach1_context_path = "/export/data2/chuber/2024/Whisper+Memory/CL/qwen0/data"
approach2_context_path = "/export/data2/chuber/2024/Whisper+Memory/CL/cb_only_memoryfiles1/hypos"

def load_hypos(path):
    id2hypo = {}
    for file in glob(path+"/*.hyp*"):
        for line in open(file):
            line = line.strip().split()
            id = line[0]
            hypo = " ".join(line[1:])
            id2hypo[id] = hypo
    return id2hypo

id2approach2_context_hypo = load_hypos(approach2_context_path)

orderfile = "/export/data2/chuber/2024/CSData/memory_files/order_0.txt"

for line in open(orderfile):
    id = line.strip()

    for set_ in ["train","dev"]:
        segfile = f"{approach1_context_path}/{id}.{set_}.seg.aligned"
        hypofile = f"{approach1_context_path}/{id}.{set_}.ref"
        newwordfile = f"{approach1_context_path}/{id}.{set_}.new_words"
        #print(segfile)

        for seg,hypo,newwords in zip(open(segfile),open(hypofile),open(newwordfile)):
            seg = seg.strip()
            hypo = hypo.strip()
            newwords = newwords.strip()

            id_ = seg.split()[0]
            approach2_context_hypo = id2approach2_context_hypo[id_]

            different_newwords = []
            for newword in newwords.split("|"):
                if newword.lower() in approach2_context_hypo.lower():
                    continue
                
                different_newwords.append(newword)

            if len(different_newwords) > 0:
                #print("Approach 1 Context Hypo:",hypo)
                #print("Approach 2 Context Hypo:",approach2_context_hypo)
                #print("New words:",different_newwords)
                for newword in different_newwords:
                    print(newword)
