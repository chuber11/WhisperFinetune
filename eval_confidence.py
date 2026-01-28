
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

correct_scores = []
correct_scores_ne = []
incorrect_scores_ne = []
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
        if label == 0:
            correct_scores.append(score[0])
        else:
            pass #print("WARNING")
    for word, score, label in suffix:
        if label == 0:
            correct_scores.append(score[0])
        else:
            pass #print("WARNING")

    for item in data:
        for word, score, label in item:
            if label == 1:
                correct_scores_ne.append(score[1])
            else:
                incorrect_scores_ne.append(score[2])
            #if label != 0:
            #    correct_scores_ne.append(score[1]+score[2])

print(len(correct_scores)+len(correct_scores_ne)+len(incorrect_scores_ne))

# Calculate mean and std of correct_scores
mean = sum(correct_scores) / len(correct_scores)
std = (sum((x - mean) ** 2 for x in correct_scores) / len(correct_scores)) ** 0.5
print(f"Mean:              {100*mean:.2f}, Std:              {100*std:.2f}")

# Calculate mean and std of correct_scores_ne
mean_ne = sum(correct_scores_ne) / len(correct_scores_ne)
std_ne = (sum((x - mean_ne) ** 2 for x in correct_scores_ne) / len(correct_scores_ne)) ** 0.5
print(f"Mean NE:           {100*mean_ne:.2f}, Std NE:           {100*std_ne:.2f}")

# Calculate mean and std of incorrect_scores_ne
mean_incorrect_ne = sum(incorrect_scores_ne) / len(incorrect_scores_ne)
std_incorrect_ne = (sum((x - mean_incorrect_ne) ** 2 for x in incorrect_scores_ne) / len(incorrect_scores_ne)) ** 0.5
print(f"Mean Incorrect NE: {100*mean_incorrect_ne:.2f}, Std Incorrect NE: {100*std_incorrect_ne:.2f}")
