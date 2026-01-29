
import soundfile as sf
from transformers import AutoModel, AutoProcessor

# Load the model and processor
model = AutoModel.from_pretrained(
    "/home/jniehues/code/LLM-LT-pipeline/qwen-omni-vllm/models/Qwen2.5-Omni-7B",
    dtype="auto",
    device_map="auto",
    enable_audio_output=False
)
processor = AutoProcessor.from_pretrained("/home/jniehues/code/LLM-LT-pipeline/qwen-omni-vllm/models/Qwen2.5-Omni-7B")

# Prepare the conversation history
conversations = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "/project/OML/chuber/2025/LT2.0/Voxtral/faster-whisper/oneminute.mp3"},
            {"type": "text", "text": "Transcribe the given audio."}
        ],
    },
]

# Process the inputs
inputs = processor(conversations, return_tensors="pt", padding=True)

# Generate the response
text_ids = model.generate(**inputs)

# Decode the text response
response_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("Response:", response_text[0])


