
from glob import glob
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
from tqdm import tqdm
from dataclasses import dataclass
import os
import sys

user = sys.argv[1]

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens), device=emission.device)
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def segment(input_values, transcription, start, end, fast=False):
    if fast:
        length = sum(len(t) for t in transcription)
        res = []
        s = start
        for t in transcription:
            e = s+(end-start)*len(t)/length
            res.append([s,e])
            s = e
        return res

    input_values = input_values[:,int(16000*start):int(16000*end)]
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
        emission = torch.log_softmax(logits[0], dim=-1)

    # Define the vocabulary and prepare token list
    vocab_dict = processor.tokenizer.get_vocab()
    char_list = [x[0] for x in sorted(vocab_dict.items(), key=lambda item: item[1])]

    transcription = ["".join(c if c in char_list or c==" " else "~" for c in t.strip().upper()) for t in transcription]
    transcript = "|" + "|".join(transcription).replace(" ","|") + "|"

    tokens = torch.as_tensor([vocab_dict[c] if c!="~" else vocab_dict["<unk>"] for c in transcript],device=emission.device)

    trellis = get_trellis(emission, tokens)
    trellis, emission, tokens = trellis.cpu(), emission.cpu(), tokens.cpu()
    path = backtrack(trellis, emission, tokens)
    segments = merge_repeats(path, transcript)

    res = []
    i = 1
    while True:
        s = None
        while transcription[0]:
            if s is None:
                s = segments[i].start
            e = segments[i].end
            transcription[0] = transcription[0][1:]
            i += 1

        res.append([start + (end-start)*s/emission.shape[0],start + (end-start)*e/emission.shape[0]])
        i += 1
        transcription.pop(0)
        if not transcription:
            break

    return res

def my_split(text, split_on):
    output = [""]
    i = 0
    while i < len(text):
        if any(text[i:i+2]==s+" " for s in split_on):
            output[-1] += text[i:i+1]
            output.append("")
            i += 1
        else:
            output[-1] += text[i]
        i += 1
    return output

device = "cuda"

# Load pre-trained model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
if device == "cuda":
    model = model.cuda()

for hypo in tqdm(glob(f"data_to_process_{user}/*.hypo.json")):
    id = hypo[:-len("hypo.json")]

    if os.path.isfile(f"data_processed_{user}/{id}seg.aligned"):
        continue

    mp3 = id + "mp3"

    # Load your audio file
    speech_array, sampling_rate = torchaudio.load(mp3)
    # Ensure the audio file is at the correct sampling rate
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech = resampler(speech_array).squeeze().numpy()

    # Tokenize the input
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    if device == "cuda":
        input_values = input_values.cuda()

    data = json.load(open(f"{id}hypo.json"))
    
    segments = json.load(open(hypo))["segments"]
    whole_transcript = "".join(s["text"] for s in segments).strip()

    def check():
        new_whole_transcript = " ".join(s[2] for s in new_segments)
        return whole_transcript == new_whole_transcript

    split_on = [".","?","!"]
    offset = 0.1

    new_segments = []
    cont = False
    i_segment = 0
    start = None
    transcript = ""
    while True:
        if i_segment == len(segments):
            break
        if start is None:
            start = segments[i_segment]["start"]

        end = segments[i_segment]["end"]
        if transcript:
            transcript += " "
        transcript += segments[i_segment]["text"].strip()

        i_segment += 1

        if i_segment == len(segments):
            sentences = [transcript]
            try:
                sentence_segments = segment(input_values, sentences, start, end)
            except Exception as e:
                print("NOOOOO",id)
                cont = True
                break

            start = sentence_segments[0][0]-offset
            end = sentence_segments[0][1]+offset
            new_segments.append([start,end,transcript])
            transcript = ""
            start = None
            #check()

            if cont:
                break
            continue

        sentences = my_split(transcript,split_on)
        if len(sentences) <= 1:
            continue

        try:
            sentence_segments = segment(input_values, sentences, start, end)
        except Exception as e:
            print("NOOOOO2",id)
            cont = True
            break

        for s1,s2,sentence in zip(sentence_segments[:-1],sentence_segments[1:],sentences[:-1]):
            start = s1[0]-offset
            end = s1[1]+offset
            new_segments.append([start,end,sentence])

        start = sentence_segments[-1][0]
        transcript = sentences[-1]
        #check()

    if cont:
        continue

    if not check():
        print("CHECK FAILED")
        continue

    min_time = 0

    index = 0
    with open(f"{id}seg.aligned","w") as f, open(f"{id}baseline.hypo","w") as f2:
        start, end, trans = None, None, ""
        n_s = new_segments
        for i,(s, e, t) in enumerate(n_s):
            if start is None:
                start = s
            if trans is not None:
                trans += " "
            trans += t
            end = e
            if i==len(n_s)-1 or end-start > min_time or n_s[i+1][0]!=e:
                f.write(f"{id[len(f'data_to_process_{user}/'):-1]}_{index} LT_CL/data_processed_{user}/{id[len(f'data_to_process_{user}/'):]}mp3 {start:.2f} {end:.2f}\n")
                f2.write(f"{trans.strip()}\n")
                index += 1
                start = None
                trans = ""
