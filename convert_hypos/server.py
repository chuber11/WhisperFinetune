
from convert import run_llm
from flask import Flask, request
import logging
from openai import OpenAI
import getpass
from load_save import load_dict, save_dict
import threading

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

app = Flask(__name__)

outputs = load_dict("outputs.pt")

cost = (2.5,10) # Cost for 1M input/output tokens for chosen model
current_cost = 0
max_cost = 1 # $

cost_lock = threading.Lock()

@app.route("/predictions/asr-numbers", methods=["POST"])
def inference():
    global current_cost
    text = request.form.get("text", "")
    unstable = request.form.get("unstable", "True") == "True"
    if not unstable and not dummy:
        print(f"Doing request with {text = }")
        run = True
        with cost_lock:
            if current_cost >= max_cost:
                run = False
                print("Not running because of cost!")
        if run:
            text_out, used_tokens = run_llm(text, model="gpt-4o-2024-08-06", max_tokens=512, number=0) # used tokens: {'completion_tokens': 16, 'prompt_tokens': 568, 'total_tokens': 584}
            if text and text[-1] != "." and text_out and text_out[-1] == ".":
                text_out = text_out[:-1]
            text = " "+text_out
            with cost_lock:
                current_cost += cost[0]*used_tokens['prompt_tokens']/1000000 + cost[1]*used_tokens['completion_tokens']/1000000
                print(f"New text: {text}, {current_cost = :.2f}$")
    elif not unstable:
        print(text,flush=True)

    return text, 200

if __name__ == "__main__":
    dummy = input("Dummy? (Y/n) ")
    dummy = dummy != "n"

    print(f"{dummy = }")

    app.run(host="0.0.0.0", port=4998)
