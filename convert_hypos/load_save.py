
import os
import torch
import threading

lock = threading.Lock()

def load_dict(name="embeddings_transcripts.pt"):
    if os.path.isfile(name):
        return torch.load(name)
    else:
        return {}

def save_dict(d, name="embeddings_transcripts.pt"):
    with lock:
        torch.save(d, name+"_new")
        os.rename(name+"_new",name)

