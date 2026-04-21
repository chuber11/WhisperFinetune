
# app.py
import os
import math
import numpy as np
import soundfile as sf
import gradio as gr

import subprocess
import requests

from model import WhisperForConditionalGenerationConfidence
from transformers import WhisperProcessor
import torch

def load_model(model_name="openai/whisper-large-v2"):
    model_name2 = "saves/model_confidence_large-v2-cont/checkpoint-8000/full"
    model = WhisperForConditionalGenerationConfidence.from_pretrained(model_name2, torch_dtype="auto", device_map="cuda")

    processor = WhisperProcessor.from_pretrained(model_name)

    return {"model":model,"processor":processor}

def load_features(file, text):
    processor = model["processor"]
    tokenizer = processor.tokenizer
    tokenizer.set_prefix_tokens(language="english", task="transcribe")

    audio, sr = sf.read(file)
    assert sr == 16000
    audio_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )
    text_features = tokenizer(text, return_tensors="pt")

    input_ids = text_features["input_ids"].cuda()
    inputs = {"input_features":audio_features["input_features"].cuda(), "labels":input_ids[:,1:], "decoder_input_ids":input_ids[:,:-1]}

    return inputs

def process_audio(audio_filepath):
    tokenizer = model["processor"].tokenizer

    command = f"ffmpeg -y -i {audio_filepath} -ac 1 -ar 16000 /tmp/test.wav"
    subprocess.run(command.split())

    wav = open("/tmp/test.wav","rb").read()
    transcription = requests.post("http://192.168.0.60:5008/asr/infer/None,None", files={"pcm_s16le":wav, "prefix":""}).json()["hypo"]

    inputs = load_features("/tmp/test.wav", transcription)

    with torch.no_grad():
        output = model["model"](**inputs)
        scores = torch.nn.functional.softmax(output.decoder_hidden_states,-1).cpu().tolist()

    res = []
    current_word = []
    last_scores_ = None
    for token_id,scores_ in zip(inputs["labels"][0].tolist()[3:-1],scores[0][3:-1]):
        converted_token = tokenizer.convert_ids_to_tokens(token_id)

        if current_word and (converted_token.startswith("Ġ") or converted_token in [".",",","?","!",";",":"]):
            res_ = [tokenizer.decode(current_word)]
            for score in last_scores_[:1]:
                res_.append(f"{100*score:.1f}%")
            res.append(res_)
            current_word = [token_id]
        else:
            current_word.append(token_id)
        last_scores_ = scores_
    if current_word:
        res_ = [tokenizer.decode(current_word)]
        for score in last_scores_[:1]:
            res_.append(f"{100*score:.1f}%")
        res.append(res_)

    return res

with gr.Blocks(title="Audio Recorder") as demo:
    gr.Markdown("# 🎤 Record audio and get confidence estimation of named entities")

    audio_in = gr.Audio(
        type="filepath",
        label="Input audio",
    )

    process_btn = gr.Button("Process")
    #output = gr.Dataframe(headers=["Token", "Prob. no NE", "Prob. correct NE", "Prob. incorrect NE"],label="Response")
    output = gr.Dataframe(headers=["Token", "Confidence"],label="Response")

    process_btn.click(process_audio, audio_in, output)

if __name__ == "__main__":
    model = load_model()

    demo.launch(server_name="0.0.0.0")

