
import torch

from data import MyDataset, DataCollatorSpeechSeq2SeqWithPadding, ConcatDataset

from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast, AutoModel
from model import WhisperForConditionalGenerationMemory, WhisperForConditionalGenerationConfidence

from torch.cuda.amp import autocast

from tqdm import tqdm

import sys
from glob import glob
import os

import argparse

from peft import PeftModel

import re
import json

parser = argparse.ArgumentParser()

parser.add_argument('--segfiles', type=str, nargs="+", help='Data used for training', default=["data_test/cv.*.test.seg.aligned"])
parser.add_argument('--decode_only_first_part', action="store_true")
parser.add_argument('--model_path', type=str, help='Path to store the trained model', default="./saves/model_newwords3/checkpoint-44000")
parser.add_argument('--use_memory', action="store_true")
parser.add_argument('--eval_confidence', action="store_true")
parser.add_argument('--memory_file', type=str)
parser.add_argument('--memory_num_distractors', type=int, default=0)
parser.add_argument('--model_name', type=str, help='Model architecture to train', default="openai/whisper-large-v2")
parser.add_argument('--hypo_file', type=str, help='Where to write the hypo')
parser.add_argument('--num_beams', type=int, default=4)
parser.add_argument('--load_adapter_model', type=str, help='Load adapter weights for baseline weights', default=None)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--no_write_at_end', action="store_true")
parser.add_argument('--language', type=str, default="english")
parser.add_argument('--force_exact_memory', type=int, default=0)

args = parser.parse_args()

if args.hypo_file is None:
    args.hypo_file = f"hypos/hypo_{args.model_path.replace('/','_')}.txt"

print(args)

outputfile = args.hypo_file
if os.path.isfile(outputfile) and os.path.getsize(outputfile) > 0:
    print(f"Output {outputfile} file already exists, exiting!")
    sys.exit()

if len(args.segfiles) == 1:
    dataset = MyDataset(args.segfiles[0], test=True, dev=args.decode_only_first_part, confidence=args.eval_confidence)
else:
    dataset = ConcatDataset([MyDataset(s, test=True, dev=args.decode_only_first_part, confidence=args.eval_confidence) for s in args.segfiles])

tokenizer = WhisperTokenizerFast.from_pretrained(args.model_name)
tokenizer.set_prefix_tokens(language=args.language, task="transcribe")
tokenizer.pad_token = tokenizer.eos_token

processor = WhisperProcessor.from_pretrained(args.model_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer, return_ids=True)

if args.use_memory:
    model_class = WhisperForConditionalGenerationMemory

    if args.memory_file is not None and args.memory_file != "None":
        prefix = " "

        def list_to_tensor(memory_words):
            double = any("->" in word for word in memory_words)
            if not double:
                memory_words = [prefix+w for w in memory_words]
            else:
                memory_words2 = []
                for w in memory_words:
                    w = w.split("->")[0]
                    memory_words2.append(prefix+w)
                for w in memory_words:
                    w = w.split("->")[-1]
                    memory_words2.append(prefix+w)
                memory_words = memory_words2
            print(memory_words)
            memory = processor.tokenizer(memory_words, return_tensors="pt", padding=True)
            memory["input_ids"] = memory["input_ids"][:,4:].cuda()
            memory["attention_mask"] = memory["attention_mask"][:,4:].cuda()
            memory["double"] = double
            memory["add_score"] = args.force_exact_memory
            return memory

        if not "data_filtered_test" in args.memory_file:
            memory_words = [line.strip() for line in open(args.memory_file)]
            memory = list_to_tensor(memory_words)
            print("MEMORY",memory_words)
        else: # for B-WER testing
            f1 = open(args.memory_file.replace("all",""))
            f2 = open(args.memory_file.replace("allwords","seg.aligned"))
            new_words = [(line2.strip().split()[0],line.strip().split("|")) for line,line2 in zip(f1,f2)]
            memory = lambda ids: [w for i,l in new_words for w in l if i in ids]
            print("MEMORY Function")
    else:
        memory = None

    new_tokens = [f"<|memory_{i}|>" for i in range(1000)]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} memory tokens")
elif args.eval_confidence:
    model_class = WhisperForConditionalGenerationConfidence
    memory = None
else:
    model_class = WhisperForConditionalGeneration
    memory = None

model = model_class.from_pretrained(args.model_path, torch_dtype="auto", device_map="cuda")

if args.load_adapter_model is not None:
    embeddings = glob(args.load_adapter_model+"/embedding.pt")
    if len(embeddings) == 1:
        embedding = torch.load(embeddings[0])
        embedding = {k[len("model."):]:v for k,v in embedding.items()}
        model.load_state_dict(embedding, strict=False)
        print("Loaded other embedding")
    elif len(embeddings) > 0:
        breakpoint()

if args.load_adapter_model is not None:
    if not os.path.isdir(args.load_adapter_model):
        print("ERROR: Adapter does not exist")
        exit()
    model = PeftModel.from_pretrained(model, args.load_adapter_model)
    model = model.merge_and_unload()

batch_size = args.batch_size

forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")

if not args.no_write_at_end:
    outputs = []
else:
    f = open(outputfile, "w")

for i in tqdm(range(0,len(dataset),batch_size)):
    data = data_collator([dataset[j] for j in range(i,min(len(dataset),i+batch_size))],inference=False)
    ids = data.pop("ids")

    #if not "7902-96595-0000" in ids:
    #    continue

    with torch.no_grad():
        if not callable(memory):
            memory_ = memory
        else:
            memory_words = memory(ids)
            l_mem = len(memory_words)
            if args.memory_num_distractors not in [0,-1]:
                num = 0
                for _,words in new_words:
                    for word in words:
                        if num >= abs(args.memory_num_distractors):
                            break
                        if word not in memory_words:
                            memory_words.append(word)
                            num += 1
                    if num >= abs(args.memory_num_distractors):
                        break
            if args.memory_num_distractors < 0:
                memory_words = memory_words[l_mem:]
            memory_ = list_to_tensor(memory_words) if memory_words else None

        text_convert = False

        if args.eval_confidence:
            data = {k:v if type(v) != torch.Tensor else v.cuda() for k,v in data.items()}
            output = model(**data)
            scores = torch.nn.functional.softmax(output.decoder_hidden_states,-1).cpu().tolist()
            transcript = []
            for score, labels, clabels in zip(scores, data["labels"], data["confidence_labels"] if "confidence_labels" in data else data["labels"]):
                # score: list of token level confidence scores
                # labels: tensor of shape (seq_len,)

                # calculate word level confidence scores by taking the score of the last token in each word
                # and append a list of (word, confidence_score) tuples to transcript
                word_confidences = []
                current_word = []
                for token_id, token_score, clabel in zip(labels.cpu().tolist()[3:], score[2:], clabels.cpu().tolist()[2:]):  # skip the first 3 tokens (they are special tokens)
                    if token_id in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                        if current_word:
                            word = tokenizer.decode(current_word)
                            word_confidences.append((word, token_score, clabel))
                            current_word = []
                        break
                    converted_token = tokenizer.convert_ids_to_tokens(token_id)
                    if current_word and (converted_token.startswith("Ġ") or converted_token in [".",",","?","!",";",":"]):
                        word = tokenizer.decode(current_word)
                        word_confidences.append((word, token_score, clabel))
                        current_word = [token_id]
                    else:
                        current_word.append(token_id)
                if current_word:
                    word = tokenizer.decode(current_word)
                    word_confidences.append((word, token_score, clabel))
                transcript.append(json.dumps(word_confidences))
        elif not text_convert:
            input_features = data["input_features"].cuda()
            model.generation_config.suppress_tokens = [t for t in model.generation_config.suppress_tokens if t!=25] # allow for : to be decoded
            kwargs = {}
            if args.use_memory:
                kwargs["first_memory_id"] = data["first_memory_id"]
            transcript = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, no_repeat_ngram_size=6, num_beams=args.num_beams, memory=memory_, **kwargs)
            transcript = tokenizer.batch_decode(transcript, skip_special_tokens=True)

            memory_map = {str(i):prefix+(w if not "->" in w else w.split("->")[1]) for i,w in enumerate(memory_words)}
            pattern = r"<\|memory_(\d+)\|>"
            def replacer(match):
                key = match.group(1)  # the number inside memory_{i}
                return memory_map.get(key, f"<missing:{key}>")

            transcript = [re.sub(pattern, replacer, t) for t in transcript]
        else:
            #given_hypofile = args.hypo_file.replace("replacements_text_filtered_other.","").replace("replacements_text_filtered.","")
            given_hypofile = args.hypo_file.replace("replacements_text_other.","").replace("oracle_text.","").replace("replacements_other_plus_text.","replacements_other.")

            given_hypos = {words[0]:" ".join(words[1:]) for line in open(given_hypofile) if (words := line.strip().split())}
            transcript = [given_hypos[i] for i in ids]

            for word in memory_words:
                if not "->" in word:
                    continue
                w1, w2 = word.split("->")
                transcript = [re.sub(rf"\b{re.escape(w1)}\b", w2, t) for t in transcript]

    for t,id in zip(transcript, ids):
        print(id,t)
        if type(t) is str:
            t = t.replace("\n"," ")
        if not args.no_write_at_end:
            outputs.append(id+" "+t+"\n")
        else:
            f.write(id+" "+t+"\n")

if not args.no_write_at_end:
    with open(outputfile, "w") as f:
        for o in outputs:
            f.write(o)

