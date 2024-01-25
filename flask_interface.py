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
from transformers.modeling_outputs import BaseModelOutput

host = "0.0.0.0"
port = 5000

app = Flask(__name__)

def create_unique_list(my_list):
    my_list = list(set(my_list))
    return my_list

def initialize_model():
    filename = "large-v3" # ["tiny.en","tiny","base.en","base","small.en","small","medium.en","medium","large-v1","large-v2","large"]
    #filename = "tiny"
    
    model_path = "openai/whisper-{}".format(filename)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    model_path = "saves/model5/checkpoint-180"
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    print("ASR initialized")

    max_batch_size = 8

    if torch.cuda.is_available():
        model = model.cuda()
    
    return (model, processor), max_batch_size

def add_prefix_tokens(processor, prefix, forced_decoder_ids):
    if len(prefix) > 0:
        prompt_ids = processor.get_prompt_ids(prefix).tolist()[1:]
        for wid in prompt_ids:
            forced_decoder_ids.append((len(forced_decoder_ids) + 1, wid))

def infer_batch(audio_wavs, prefix="", input_language="en", task="transcribe", audio_sample_rate=16000):
    # get device based on the model parameters
    device = next(model.parameters()).device
        
    input_values = torch.cat([processor(item, sampling_rate=audio_sample_rate, return_tensors="pt").input_features for item in audio_wavs], dim=0).to(device)
    
    if input_language != "None" and not "+" in input_language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=input_language, task=task)
        add_prefix_tokens(processor, prefix, forced_decoder_ids)
    else:
        output = model.generate(
            input_values, 
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        ) # Predicts the language

        if "+" not in input_language:
            predicted_ids = output.sequences[:,1]
        else:
            input_languages = input_language.split("+")
            predictable_ids = [id for id,token in processor.tokenizer.added_tokens_decoder.items() if token.content[2:-2] in input_languages]

            predicted_ids_small = output.scores[0][:,predictable_ids].argmax(-1)
            predicted_ids = torch.as_tensor(predictable_ids, device=predicted_ids_small.device)[predicted_ids_small]

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task=task)
        add_prefix_tokens(processor, prefix, forced_decoder_ids)

        pred_to_indices = {}
        for i,pred in enumerate(predicted_ids.tolist()):
            if pred not in pred_to_indices:
                pred_to_indices[pred] = [i]
            else:
                pred_to_indices[pred].append(i)

        outputs = {}
        for pred, indices in pred_to_indices.items():
            forced_decoder_ids[0] = (forced_decoder_ids[0][0],pred)
            encoder_outputs = BaseModelOutput(last_hidden_state=output["encoder_hidden_states"][-1][indices])

            predicted_ids2 = model.generate(
                input_values[indices], 
                forced_decoder_ids=forced_decoder_ids,
                no_repeat_ngram_size=6,
                encoder_outputs=encoder_outputs,
            )
            for o,i in zip(processor.batch_decode(predicted_ids2, skip_special_tokens=True),indices):
                outputs[i] = o
        
        return [outputs[i] for i in range(len(outputs))]

    predicted_ids = model.generate(
        input_values, 
        forced_decoder_ids=forced_decoder_ids,
        no_repeat_ngram_size=6
    )

    text_output_raw = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return text_output_raw

def use_model(reqs):

    if len(reqs) == 1:
        req = reqs[0]
        audio_tensor, prefix, input_language, output_language = req.get_data()
        if not (input_language == output_language or output_language == 'en'):
            result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(input_language, output_language)}
            req.publish(result)
            return
        
        if input_language == output_language:
            task = "transcribe"
        else:
            task = "translate"
        
        hypo = infer_batch(audio_wavs=[audio_tensor], input_language=input_language, task=task, prefix=prefix)[0]
            
        result = {"hypo": hypo.strip()}
        req.publish(result)

    else:
        audio_tensors = list()
        prefixes = ['']
        input_languages = list()
        output_languages = list()

        batch_runnable = False

        for req in reqs:
            audio_tensor, prefix, input_language, output_language = req.get_data()
            audio_tensors.append(audio_tensor)
            prefixes.append(prefix)
            input_languages.append(input_language)
            output_languages.append(output_language)

        unique_prefix_list = create_unique_list(prefixes)
        unique_input_languages = create_unique_list(input_languages)
        unique_output_languages = create_unique_list(output_languages)
        if len(unique_prefix_list) == 1 and len(unique_input_languages) == 1 and len(unique_output_languages) == 1:
            batch_runnable = True

        if batch_runnable:
            if unique_input_languages[0] == unique_output_languages[0]:
                task = "transcribe"
            elif unique_output_languages[0] == 'en':
                task = "translate"
            else:
                for req in reqs:
                    result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(unique_input_languages[0], unique_output_languages[0])}
                    req.publish(result)
                return
            hypos = infer_batch(audio_wavs=audio_tensors, input_language=input_languages[0], task=task, prefix=prefixes[0])

            for req, hypo in zip(reqs, hypos):
                result = {"hypo": hypo.strip()}
                req.publish(result)
        else:
            for req, audio_tensor, prefix, input_language, output_language \
                    in zip(reqs, audio_tensors, prefixes[1:], input_languages, output_languages):
                if not (input_language == output_language or output_language == 'en'):
                    result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(input_language, output_language)}
                    req.publish(result)
                else:
                    
                    if input_language == output_language:
                        task = "transcribe"
                    else:
                        task = "translate"
                    hypo = infer_batch(audio_wavs=[audio_tensor], input_language=input_language, task=task, prefix=prefix)[0]                    
                    result = {"hypo": hypo.strip()}
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
        data = (audio_tensor,prefix,input_language,output_language)

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

(model, processor), max_batch_size = initialize_model()

queue_in = queue.PriorityQueue()
dict_out = {}

decoding = threading.Thread(target=run_decoding)
decoding.daemon = True
decoding.start()


if __name__ == "__main__":
    app.run(host=host, port=port)

