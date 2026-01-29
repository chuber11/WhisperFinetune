
import json
from tqdm import tqdm
import sys

n = sys.argv[1]

id_to_data = {}
for line in open(f"hypos_confidence/dev{n}.txt"):
    id, data = line.strip().split(maxsplit=1)
    data = json.loads(data)

    # data: list of (word, confidence_score, confidence_label) tuples

    if id not in id_to_data:
        id_to_data[id] = []
    id_to_data[id].append(data)

confusion_matrix = [[0 for _ in range(3)] for _ in range(3)]
for id, data in tqdm(id_to_data.items()):
    # for one id the prefix and suffix of all the list items are the same
    # extract prefix and suffix
    prefix = []
    suffix = []
    while True:
        if not data:
            break
        first_item = data[0]
        if not first_item:
            break
        all_same = all(item[0][0] == first_item[0][0] for item in data if item)
        if all_same:
            prefix.append(first_item[0])
            for item in data:
                item.pop(0)
        else:
            break
    while True:
        if not data:
            break
        first_item = data[0]
        if not first_item:
            break
        all_same = all(item[-1][0] == first_item[-1][0] for item in data)
        if all_same:
            suffix.insert(0, first_item[-1])
            for item in data:
                item.pop(-1)
        else:
            break

    for word, score, label in prefix:
        argmax = max(range(len(score)), key=score.__getitem__)
        confusion_matrix[label][argmax] += 1
    for word, score, label in suffix:
        argmax = max(range(len(score)), key=score.__getitem__)
        confusion_matrix[label][argmax] += 1

    for item in data:
        for word, score, label in item:
            argmax = max(range(len(score)), key=score.__getitem__)
            confusion_matrix[label][argmax] += 1

# normalize confusion matrix
for i in range(3):
    row_sum = sum(confusion_matrix[i])
    if row_sum > 0:
        confusion_matrix[i] = [x / row_sum for x in confusion_matrix[i]]

print("Confusion Matrix:")
for row in confusion_matrix:
    print("\t".join(map(lambda v:f"{100*v:4.1f}", row)))