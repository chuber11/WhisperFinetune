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
import requests
from glob import glob

from model import BaseModelOutputMemory
from model import WhisperForConditionalGenerationMemory

host = "0.0.0.0"
port = 4999

app = Flask(__name__)

initialize_output = {}

def get_latest_adaptation_path(user):
    peft_model = None
    peft_dirs = sorted(glob(f"saves/model_CL_{user}_*"))
    if len(peft_dirs) > 0:
        peft_dir = glob(f"{peft_dirs[-1]}/checkpoint-*")
        if len(peft_dir) > 0:
            peft_model = peft_dir[0]
    return peft_model

def initialize(user=None):
    if "model" in initialize_output:
        model = initialize_output.pop("model")
        del model

    #filename = "large-v2" # ["tiny.en","tiny","base.en","base","small.en","small","medium.en","medium","large-v1","large-v2","large"]
    filename = "saves/model_newwords15/checkpoint-184000"
    #filename = "tiny"
    
    #model_path = "openai/whisper-{}".format(filename)
    model_path = filename
    processor = WhisperProcessor.from_pretrained(model_path)

    processor.get_decoder_prompt_ids(language="en", task="transcribe") # WARNING: Changes state of processor
    
    model_path = "saves/model_newwords15/checkpoint-184000"
    #model_path = "saves/model_newwords15_2/checkpoint-5000"

    model = WhisperForConditionalGenerationMemory.from_pretrained(model_path)
    #model = WhisperForConditionalGeneration.from_pretrained(model_path)

    initialize_output["peft_model"] = None

    if user is not None:
        peft_model = get_latest_adaptation_path(user)

        if peft_model is not None:
            print(f"Loading factorization model {peft_model} for {user = }...")

            from peft import PeftModel
            model = PeftModel.from_pretrained(model, peft_model)
            model = model.merge_and_unload()

        initialize_output["peft_model"] = peft_model

    model.generation_config.suppress_tokens = [t for t in model.generation_config.suppress_tokens if t!=25] # allow for : to be decoded
    model.generation_config.begin_suppress_tokens.remove(50257)
    model.generation_config.lang_to_id = {i:i for i in range(50259,50358)}
    
    max_batch_size = 1

    if torch.cuda.is_available():
        model = model.cuda()
    
    print("ASR initialized")

    initialize_output["user"] = user
    initialize_output["max_batch_size"] = max_batch_size
    initialize_output["model"] = model
    initialize_output["processor"] = processor

def get_forced_decoder_ids(processor, prefix, task, device, force_language):
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task=task) # <|en|><|transcribe|><|notimestamps|>
    if force_language == 0:
        forced_decoder_ids[0] = (forced_decoder_ids[0][0],None) # Unset language
    if len(prefix) > 0:
        ids = processor.get_prompt_ids(prefix).tolist()[1:]
        for id in ids:
            forced_decoder_ids.append((len(forced_decoder_ids) + 1, id))

    #print(forced_decoder_ids)
    return forced_decoder_ids

def infer_batch(audio_wavs, prefix="", input_language="en", task="transcribe", audio_sample_rate=16000, memory_words=None, num_beams=4, force_language=0):
    model = initialize_output["model"]
    processor = initialize_output["processor"]

    # get device based on the model parameters
    device = next(model.parameters()).device
        
    input_values = torch.cat([processor(item, sampling_rate=audio_sample_rate, return_tensors="pt").input_features for item in audio_wavs], dim=0).to(device)

    memory_prefix = " "

    if memory_words is not None and len(memory_words) > 0:
        double = True #any("->" in w for w in memory_words)
        if not double:
            memory_words = [memory_prefix+w for w in memory_words]
        else:
            memory_words2 = []
            for w in memory_words:
                w = w.split("->")[0]
                memory_words2.append(memory_prefix+w)
            for w in memory_words:
                w = w.split("->")[-1]
                memory_words2.append(memory_prefix+w)
            memory_words = memory_words2
        memory = processor.tokenizer(memory_words, return_tensors="pt", padding=True)
        memory["input_ids"] = memory["input_ids"][:,4:].to(device)
        memory["attention_mask"] = memory["attention_mask"][:,4:].to(device)
        memory["double"] = double
        memory["add_score"] = 25
        #print([[processor.tokenizer.decode(i) for i in memory["input_ids"][j]] for j in range(len(memory["input_ids"]))])
        #print(memory["attention_mask"])
        print(memory_words)
    else:
        memory = None

    forced_decoder_ids = get_forced_decoder_ids(processor, prefix, task, device, force_language)
    model.generation_config.forced_decoder_ids = forced_decoder_ids

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
    num_beamss = list()
    force_languages = list()

    batch_runnable = False

    for req in reqs:
        audio_tensor, prefix, input_language, output_language, memory_words, user, num_beams, force_language = req.get_data()
        audio_tensors.append(audio_tensor)
        prefixes.append(prefix)
        input_languages.append(input_language)
        output_languages.append(output_language)
        memory_wordss.append(tuple(memory_words) if memory_words is not None else memory_words)
        num_beamss.append(num_beams)
        force_languages.append(force_language)

        if user != initialize_output["user"] or get_latest_adaptation_path(user) != initialize_output["peft_model"]:
            initialize(user)

    if len(set(prefixes)) == 1 and len(set(input_languages)) == 1 and len(set(output_languages)) == 1 and len(set(memory_wordss)) == 1 and len(set(num_beamss)) == 1 and len(set(force_languages)) == 1:
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
        hypos, lids = infer_batch(audio_wavs=audio_tensors, input_language=input_languages[0], task=task, prefix=prefixes[0], memory_words=memory_wordss[0], num_beams=num_beamss[0], force_language=force_languages[0])

        for req, hypo, lid in zip(reqs, hypos, lids):
            result = {"hypo": hypo.strip(), "lid":lid}
            req.publish(result)
    else:
        for req, audio_tensor, prefix, input_language, output_language, memory_words, num_beams, force_language \
                in zip(reqs, audio_tensors, prefixes, input_languages, output_languages, memory_wordss, num_beamss, force_languages):
            if input_language == output_language:
                task = "transcribe"
            elif output_language == 'en':
                task = "translate"
            else:
                result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(input_language, output_language)}
                req.publish(result)
                continue
            hypo, lid = infer_batch(audio_wavs=[audio_tensor], input_language=input_language, task=task, prefix=prefix, memory_words=memory_words, num_beams=num_beams, force_language=force_language)
            result = {"hypo": hypo[0].strip(), "lid":lid[0]}
            req.publish(result)

def run_decoding():
    while True:
        reqs = [queue_in.get()]
        while not queue_in.empty() and len(reqs) < initialize_output["max_batch_size"]:
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
        
        if memory is not None:
            memory_individual_words = True
            if memory_individual_words:
                memory = [word for it in memory for word in it.split()]
            memory = sorted(list(set(memory)))
        print(f"{memory = }")
        
    user = request.files.get("user") # can be None
    if user is not None:
        user: str = user.read().decode("utf-8")

    num_beams = request.files.get("num_beams") # can be None
    try:
        num_beams = int(num_beams.read()) # used together with priority queue
    except:
        num_beams = 1

    force_language = request.files.get("force_language") # can be None
    try:
        force_language = int(force_language.read()) # used together with priority queue
    except:
        force_language = 0

    priority = request.files.get("priority") # can be None
    try:
        priority = int(priority.read()) # used together with priority queue
    except:
        priority = 0

    run_baseline = initialize_output["peft_model"]

    if run_baseline:
        result_queue = queue.Queue()
        def run():
            res = requests.post(f"http://192.168.0.60:5008/asr/infer/{input_language},{output_language}", files={"pcm_s16le":pcm_s16le, "prefix":prefix, "priority":priority})
            if res.status_code != 200:
                result_queue.put({"hypo": "", "status":res.status_code})
            result_queue.put(res.json())

        t = threading.Thread(target=run)
        t.start()

    # calculate features corresponding to a torchaudio.load(filepath) call
    audio_tensor = pcm_s16le_to_tensor(pcm_s16le).squeeze()

    condition = threading.Condition()
    with condition:
        id = str(uuid.uuid4())
        data = (audio_tensor,prefix,input_language,output_language,memory,user, num_beams, force_language)

        queue_in.put(Priority(priority,id,condition,data))

        condition.wait()

    result = dict_out.pop(id)
    status = 200
    if status in result:
        status = result.pop(status)

    if run_baseline:
        t.join()
        result_baseline = result_queue.get()

        if "lid" in result_baseline and result_baseline["lid"] not in ["en","de"]:
            print("Falling back to baseline model because detected language is not English or German!")
            result = result_baseline

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

initialize()

queue_in = queue.PriorityQueue()
dict_out = {}

decoding = threading.Thread(target=run_decoding)
decoding.daemon = True
decoding.start()


if __name__ == "__main__":
    app.run(host=host, port=port)

