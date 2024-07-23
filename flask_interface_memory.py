# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from flask import Flask, request
import torch
import numpy as np
import math
import sys
import json
import threading
import queue
import uuid
import traceback
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from model import BaseModelOutputMemory
from model import WhisperForConditionalGenerationMemory

host = "0.0.0.0"
port = 4999

app = Flask(__name__)

def initialize_model():
    filename = "large-v2" # ["tiny.en","tiny","base.en","base","small.en","small","medium.en","medium","large-v1","large-v2","large"]
    #filename = "tiny"
    
    model_path = "openai/whisper-{}".format(filename)
    processor = WhisperProcessor.from_pretrained(model_path)

    processor.get_decoder_prompt_ids(language="en", task="transcribe") # WARNING: Changes state of processor
    
    #model_path = "saves/model_newwords6/checkpoint-50000"
    #model_path = "saves/model_newwords7/checkpoint-158000"
    #model_path = "saves/model_newwords8/checkpoint-199000"
    model_path = "saves/model_newwords10/checkpoint-166000"
    #model_path = "saves/model_newwords11/checkpoint-1000"

    model = WhisperForConditionalGenerationMemory.from_pretrained(model_path)
    #model = WhisperForConditionalGeneration.from_pretrained(model_path)

    model.generation_config.suppress_tokens = [t for t in model.generation_config.suppress_tokens if t!=25] # allow for : to be decoded
    model.generation_config.begin_suppress_tokens.remove(50257)
    
    """model_path = "saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1.0003e-6_train_emb0/checkpoint-900"
    model2 = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.load_state_dict(model2.state_dict(),strict=False)
    del model2"""

    print("ASR initialized")

    max_batch_size = 8

    if torch.cuda.is_available():
        model = model.cuda()
    
    return (model, processor), max_batch_size

def get_forced_decoder_ids(processor, prefix, task, device):
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task=task) # <|en|><|transcribe|><|notimestamps|>
    forced_decoder_ids[0] = (forced_decoder_ids[0][0],None) # Unset language
    if len(prefix) > 0:
        ids = processor.get_prompt_ids(prefix).tolist()[1:]
        for id in ids:
            forced_decoder_ids.append((len(forced_decoder_ids) + 1, id))

    #print(forced_decoder_ids)
    return forced_decoder_ids

def infer_batch(audio_wavs, prefix="", input_language="en", task="transcribe", audio_sample_rate=16000, memory_words=None, num_beams=4):
    # get device based on the model parameters
    device = next(model.parameters()).device
        
    input_values = torch.cat([processor(item, sampling_rate=audio_sample_rate, return_tensors="pt").input_features for item in audio_wavs], dim=0).to(device)

    memory_prefix = " "

    if memory_words is not None and len(memory_words) > 0:
        #print(memory_words)
        memory_words = [memory_prefix+w for w in memory_words]

        memory = processor.tokenizer(memory_words, return_tensors="pt", padding=True)
        memory["input_ids"] = memory["input_ids"][:,4:].to(device)
        memory["attention_mask"] = memory["attention_mask"][:,4:].to(device)
        #print([[processor.tokenizer.decode(i) for i in memory["input_ids"][j]] for j in range(len(memory["input_ids"]))])
        #print(memory["attention_mask"])
    else:
        memory = None

    forced_decoder_ids = get_forced_decoder_ids(processor, prefix, task, device)
    model.generation_config.forced_decoder_ids = forced_decoder_ids[1:]

    predicted_ids = model.generate(
        input_values, 
        num_beams=num_beams,
        no_repeat_ngram_size=6,
        memory=memory,
    )

    text_output_raw = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    lids = [l[2:-2] for l in processor.batch_decode(predicted_ids[:,1])]

    if input_language != "None":
        input_languages = input_language.split("+")
        text_output_raw = [t if l in input_languages else prefix for t,l in zip(text_output_raw,lids)]

    return text_output_raw, lids

def use_model(reqs):
    audio_tensors = list()
    prefixes = list()
    input_languages = list()
    output_languages = list()
    memory_wordss = list()

    batch_runnable = False

    for req in reqs:
        audio_tensor, prefix, input_language, output_language, memory_words = req.get_data()
        audio_tensors.append(audio_tensor)
        prefixes.append(prefix)
        input_languages.append(input_language)
        output_languages.append(output_language)
        memory_wordss.append(tuple(memory_words) if memory_words is not None else memory_words)

    if len(set(prefixes)) == 1 and len(set(input_languages)) == 1 and len(set(output_languages)) == 1 and len(set(memory_wordss)) == 1:
        batch_runnable = True

    if batch_runnable:
        if input_languages[0] == output_languages[0]:
            task = "transcribe"
        elif output_languages[0] == 'en':
            task = "translate"
        else:
            for req in reqs:
                result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(input_languages[0], output_languages[0])}
                req.publish(result)
            return
        hypos, lids = infer_batch(audio_wavs=audio_tensors, input_language=input_languages[0], task=task, prefix=prefixes[0], memory_words=memory_wordss[0])

        for req, hypo, lid in zip(reqs, hypos, lids):
            result = {"hypo": hypo.strip(), "lid":lid}
            req.publish(result)
    else:
        for req, audio_tensor, prefix, input_language, output_language, memory_words \
                in zip(reqs, audio_tensors, prefixes, input_languages, output_languages, memory_wordss):
            if input_language == output_language:
                task = "transcribe"
            elif output_language == 'en':
                task = "translate"
            else:
                result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(input_language, output_language)}
                req.publish(result)
                continue
            hypo, lid = infer_batch(audio_wavs=[audio_tensor], input_language=input_language, task=task, prefix=prefix, memory_words=memory_words)
            result = {"hypo": hypo[0].strip(), "lid":lid[0]}
            req.publish(result)

def run_decoding():
    while True:
        reqs = [queue_in.get()]
        while not queue_in.empty() and len(reqs) < max_batch_size:
            req = queue_in.get()
            reqs.append(req)
            if req.priority >= 1:
                break

        print("Batch size:",len(reqs),"Queue size:",queue_in.qsize())

        try:
            use_model(reqs)
        except Exception as e:
            print("An error occured during model inference")
            traceback.print_exc()
            for req in reqs:
                req.publish({"hypo":"", "status":400})

class Priority:
    next_index = 0

    def __init__(self, priority, id, condition, data):
        self.index = Priority.next_index

        Priority.next_index += 1

        self.priority = priority
        self.id = id
        self.condition = condition
        self.data = data

    def __lt__(self, other):
        return (-self.priority, self.index) < (-other.priority, other.index)

    def get_data(self):
        return self.data

    def publish(self, result):
        dict_out[self.id] = result
        try:
            with self.condition:
                self.condition.notify()
        except:
            print("ERROR: Count not publish result")

def pcm_s16le_to_tensor(pcm_s16le):
    audio_tensor = np.frombuffer(pcm_s16le, dtype=np.int16)
    audio_tensor = torch.from_numpy(audio_tensor)
    audio_tensor = audio_tensor.float() / math.pow(2, 15)
    audio_tensor = audio_tensor.unsqueeze(1)  # shape: frames x 1 (1 channel)
    return audio_tensor

# corresponds to an asr_server "http://$host:$port/asr/infer/en,en" in StreamASR.py
# use None when no input- or output language should be specified
@app.route("/asr/infer/<input_language>,<output_language>", methods=["POST"])
def inference(input_language, output_language):
    pcm_s16le: bytes = request.files.get("pcm_s16le").read()
    prefix = request.files.get("prefix") # can be None
    if prefix is not None:
        prefix: str = prefix.read().decode("utf-8")

    memory = request.files.get("memory") # can be None
    if memory is not None:
        memory: list = json.loads(memory.read().decode("utf-8"))

    # calculate features corresponding to a torchaudio.load(filepath) call
    audio_tensor = pcm_s16le_to_tensor(pcm_s16le).squeeze()

    priority = request.files.get("priority") # can be None
    try:
        priority = int(priority.read()) # used together with priority queue
    except:
        priority = 0

    condition = threading.Condition()
    with condition:
        id = str(uuid.uuid4())
        data = (audio_tensor,prefix,input_language,output_language,memory)

        queue_in.put(Priority(priority,id,condition,data))

        condition.wait()

    result = dict_out.pop(id)
    status = 200
    if status in result:
        status = result.pop(status)

    # result has to contain a key "hypo" with a string as value (other optional keys are possible)
    return json.dumps(result), status

# called during automatic evaluation of the pipeline to store worker information
@app.route("/asr/version", methods=["POST"])
def version():
    # return dict or string (as first argument)
    return "Whisper", 200

@app.route("/asr/extract_words", methods=["POST"])
def extract_words_route():
    from extract_words import extract_words
    import base64
    pdf_bytes = request.files["pdf_bytes"].read()
    pdf_bytes = base64.b64decode(pdf_bytes)
    res = {"memory_words": extract_words(pdf_bytes)}
    return res, 200

    text_output_raw = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    lids = [l[2:-2] for l in processor.batch_decode(predicted_ids[:,1])]

    if input_language != "None":
        input_languages = input_language.split("+")
        text_output_raw = [t if l in input_languages else prefix for t,l in zip(text_output_raw,lids)]

    return text_output_raw, lids

@app.route("/asr/available_languages", methods=["GET","POST"])
def languages():
    return ["en","de"]

(model, processor), max_batch_size = initialize_model()

queue_in = queue.PriorityQueue()
dict_out = {}

decoding = threading.Thread(target=run_decoding)
decoding.daemon = True
decoding.start()


if __name__ == "__main__":
    app.run(host=host, port=port)

