
from convert import run_llm
from flask import Flask, request
import logging
from openai import OpenAI
import getpass
from load_save import load_dict, save_dict

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

app = Flask(__name__)

outputs = load_dict("outputs.pt")

@app.route("/predictions/mt-numbers,numbers", methods=["POST"])
def inference():
    text = request.form.get("text", "")
    unstable = request.form.get("unstable", "True") == "True"
    if not unstable and not dummy:
        print(f"Doing request with {text = }")
        text = run_llm(text, model="gpt-4o", max_tokens=512, number=0)
        print(f"New text: {text}")
    return text, 200

if __name__ == "__main__":
    dummy = input("Dummy? (Y/n) ")
    dummy = dummy != "n"

    print(f"{dummy = }")

    app.run(host="0.0.0.0", port=4998)
