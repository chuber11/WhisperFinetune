
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
from transformers import AutoProcessor
import torch
from glob import glob

model_path_encoder = "saves/model_newwords15/checkpoint-184000/model_with_prep.onnx" #sys.argv[1]
model_path_decoder = "saves/model_newwords15/checkpoint-184000/model.onnx" #sys.argv[2]

model_name = "openai/whisper-large-v2"

processor = AutoProcessor.from_pretrained(model_name)

print("Loading encoder")
session_encoder = onnx.load(model_path_encoder)
session_encoder = backend.prepare(session_encoder, device='CUDA:0')
print("Loading decoder")
session_decoder = ort.InferenceSession(model_path_decoder, providers=["CUDAExecutionProvider"])

encoder_input_names = [x.name for x in session_encoder.get_inputs()]
decoder_input_names = [x.name for x in session_decoder.get_inputs()]
encoder_output_names = [x.name for x in session_encoder.get_outputs()]
decoder_output_names = [x.name for x in session_decoder.get_outputs()]

decoder_input_ids = np.asarray([[50258, 50259, 50359, 50363]], dtype=np.int32)

sample_encoder = {"decoder_input_ids": decoder_input_ids}
for f in glob("test_*.pt"):
    key = f[len("test_"):-len(".pt")]
    value = torch.load(f, map_location="cpu").numpy()
    if key == "memory_text_mask":
        value = value.astype(np.int8)
    if key == "audio_pcm":
        value = np.expand_dims(value, axis=0)
    sample_encoder[key] = value

sample_encoder['mem_attn_out'] = sample_encoder['mem_attn_out'][:1]
sample_encoder['memory_text_enc'] = sample_encoder['memory_text_enc'][:0]
sample_encoder['memory_text_mask'] = sample_encoder['memory_text_mask'][:0]

for _ in range(5):
    print("Running encoder")
    outputs = session_encoder.run(None, sample_encoder)

    sample_decoder = {}
    output_names = encoder_output_names
    whole_transcript = []
    for i in range(100):
        for n,o in zip(output_names, outputs):
            if n.startswith("present"):
                n = n.replace("present","past")
            if n in decoder_input_names:
                sample_decoder[n] = o

        transcript = outputs[0][:,-1:].argmax(-1).astype(np.int32)
        if i+1 < decoder_input_ids.shape[1]:
            transcript[0][0] = decoder_input_ids[0,i+1]
        if transcript[0][0] == processor.tokenizer.eos_token_id:
            break
        whole_transcript.append(transcript)
        sample_decoder["input_ids"] = transcript
        for k,v in sample_encoder.items():
            if k in decoder_input_names:
                sample_decoder[k] = v

        print("Running decoder")
        outputs = session_decoder.run(None, sample_decoder)
        output_names = decoder_output_names

    whole_transcript = processor.tokenizer.decode([t[0][0] for t in whole_transcript])
    print(whole_transcript)

