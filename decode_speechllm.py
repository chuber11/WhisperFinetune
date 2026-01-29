
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

def transcribe_with_context(audio_path: str, context: str):
    """
    Perform ASR on a single audio file using optional extra text context.
    """
    # Build multimodal prompt
    prompt = [
        {"role": "system", "content": "You are a precise English ASR model."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Context: {context}"},
            {"type": "audio", "audio_url": audio_path}
        ]}
    ]
    
    # Process inputs
    inputs = processor(prompt, return_tensors="pt").to(model.device)

    # Generate output
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.0  # deterministic ASR
    )

    # Decode to text
    decoded = processor.decode(output[0], skip_special_tokens=True)
    return decoded

model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B",
                                                            torch_dtype="fp16",
                                                            device_map="auto",
                                                            attn_implementation="flash_attention_2")

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
