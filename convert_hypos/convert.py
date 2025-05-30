
from openai import OpenAI
import torch
import json
import random
import os
from tqdm import tqdm
from glob import glob
import re
import getpass
import jiwer
from load_save import load_dict, save_dict
from tqdm import tqdm

def response2output(response):
    output = json.loads(response)["choices"][0]["message"]["content"]
    usage = json.loads(response)["usage"]
    price_ = (usage['prompt_tokens']*5+usage['completion_tokens']*15)/1000000*100
    global price
    price += price_
    return output

def get_prompt(number=0):
    if number == 0:
        return "An utterance is given. Convert timestamps to a format like 18:30, years to a format like 1789, currency amounts of an english utterance to a format like $1,200.30 $1,500 €1,200.30 or €1,500, currency amounts of a german utterance to a format like 1.200,30$ 1.200$ 1.200,30€ or 1.200€ and other numbers to the same format as for currencies just without the currency symbol. If none of the before mentioned occur, output the utterance."
    elif number == 1:
        return "You are provided with a sentence which is either in english or german language. Convert timestamps to a 24 hour format like 18:30 and years to a format like 1789. If the sentence is in english language, convert currency amounts to a format like $1,200 €1,200 (for amounts without cents) and $1,200.00 €1,200.00 (for amounts with cents). For other numbers use the same format but without the currency symbol. If the sentence is in german language, convert currency amounts to a format like 1.200$ 1.200€ (for amounts without cents) and 1.200,00$ 1.200,00€ (for amounts with cents). For other numbers use the same format but without the currency symbol. If none of the before mentioned occur, output the utterance."
    elif number == 2:
        statements = (
            "You are provided with a sentence which is either in english or german language. Output the given sentence where numbers are converted to numeric literals.",
            "The peace treaty, signed in nineteen hundred and five, ended years of warfare.",
            "The peace treaty, signed in 1905, ended years of warfare.",
            "In Italy, I spent fifty euros and sixty dollars on a beautiful hand-painted vase.",
            "In Italy, I spent €50 and $60 on a beautiful hand-painted vase.",
            "Mein Bruder hat elf Dollar und zwanzig Euro in der Lotterie gewonnen.",
            "Mein Bruder hat 11$ und 20€ in der Lotterie gewonnen.",
            "The mysterious book cost me twenty-three dollars and seventy-five cents at the old bookstore downtown, which is approximately twenty-one euros and seventy-eight cents.",
            "The mysterious book cost me $23.75 at the old bookstore downtown, which is approximately €21.78.",
            "Ich habe nur zehn Euro und achtzig Cent in meiner Tasche, das sind etwa elf Dollar und achtundsiebzig Cents.",
            "Ich habe nur 10,80€ in meiner Tasche, das sind etwa 11,78$.",
            "The museum housed over ten thousand historical artifacts.",
            "The museum housed over 10,000 historical artifacts.",
            "Die Organisation spendete zehntausend Mahlzeiten an Bedürftige.",
            "Die Organisation spendete 10.000 Mahlzeiten an Bedürftige.",
            "She called me at twenty-five to eight this morning, sounding very excited.",
            "She called me at 7:35 this morning, sounding very excited.",
            "Um halb sechs am Abend machen wir Feierabend.",
            "Um 17:30 am Abend machen wir Feierabend."
            )
        return statements
    elif number == 3:
        statements = (
            "You are provided with a sentence which is either in english or german language. Convert timestamps to a 24 hour format like 18:30 and years to a format like 1789. If the sentence is in english language, convert currency amounts to a format like $1,200 €1,200 (for amounts without cents) and $1,200.00 €1,200.00 (for amounts with cents). For other numbers use the same format but without the currency symbol. If the sentence is in german language, convert currency amounts to a format like 1.200$ 1.200€ (for amounts without cents) and 1.200,00$ 1.200,00€ (for amounts with cents). For other numbers use the same format but without the currency symbol. If none of the before mentioned occur, output the utterance."
            "The peace treaty, signed in nineteen hundred and five, ended years of warfare.",
            "The peace treaty, signed in 1905, ended years of warfare.",
            "In Italy, I spent fifty euros and sixty dollars on a beautiful hand-painted vase.",
            "In Italy, I spent €50 and $60 on a beautiful hand-painted vase.",
            "Mein Bruder hat elf Dollar und zwanzig Euro in der Lotterie gewonnen.",
            "Mein Bruder hat 11$ und 20€ in der Lotterie gewonnen.",
            "The mysterious book cost me twenty-three dollars and seventy-five cents at the old bookstore downtown, which is approximately twenty-one euros and seventy-eight cents.",
            "The mysterious book cost me $23.75 at the old bookstore downtown, which is approximately €21.78.",
            "Ich habe nur zehn Euro und achtzig Cent in meiner Tasche, das sind etwa elf Dollar und achtundsiebzig Cents.",
            "Ich habe nur 10,80€ in meiner Tasche, das sind etwa 11,78$.",
            "The museum housed over ten thousand historical artifacts.",
            "The museum housed over 10,000 historical artifacts.",
            "Die Organisation spendete zehntausend Mahlzeiten an Bedürftige.",
            "Die Organisation spendete 10.000 Mahlzeiten an Bedürftige.",
            "She called me at twenty-five to eight this morning, sounding very excited.",
            "She called me at 7:35 this morning, sounding very excited.",
            "Um halb sechs am Abend machen wir Feierabend.",
            "Um 17:30 am Abend machen wir Feierabend."
            )
        return statements
    else:
        raise NotImplementedError

def run_llm(utterance, model="gpt-4-turbo-preview", max_tokens=512, number=0):
    prompt = get_prompt(number)

    if number == 0:
        key = (model, utterance)
    else:
        key = (model, utterance, prompt)

    if key not in outputs:
        global client
        if client is None:
            #client = OpenAI(api_key=getpass.getpass("Enter your api_key: "))
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        #print("Doing request", prompt, utterance)
        print("Doing request") #, prompt, utterance)

        if type(prompt) is str:
            response = client.chat.completions.create(
              model=model,
              messages=[
                {
                  "role": "system",
                  "content": prompt
                },
                {
                  "role": "user",
                  "content": utterance
                },
              ],
              temperature=1,
              max_tokens=max_tokens,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
        else:
            messages = [{"role": "system" if i==0 else "assistant" if i%2==0 else "user", "content": p} for i,p in enumerate(prompt)]+[{"role": "user", "content": utterance}]
            response = client.chat.completions.create(
              model=model,
              messages=messages,
              temperature=0,
              max_tokens=1024,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )

        response = response.model_dump_json()

        outputs[key] = response
        save_dict(outputs, "outputs_converted.pt")
    else:
        response = outputs[key]

    res = response2output(response).strip().split("\n")
    if len(res) > 1:
        print("WARNING: Result contains multiple lines! Using input as output")
        output = utterance
        return output, json.loads(response)["usage"]
    output = res[0]

    wer = jiwer.wer(output, utterance)
    if wer > 1.0:
        print(f"WARNING: WER > 1.0! Using input instead of {output = }")
        output = utterance

    return output, json.loads(response)["usage"]

price = 0
client = None
outputs = load_dict("outputs_converted.pt")

if __name__ == "__main__":
    random.seed(42)

    number = 3
    #model = "gpt-3.5-turbo"
    #model = "gpt-4-turbo-preview"
    model = "gpt-4o"
    #model = "manual"

    allfiles = [#"../hypos/hypo_openai_whisper-large-v2_beam4.*.txt",
        #"../hypos/hypo_openai_whisper-large-v2_beam4.*.human.txt",
        #"../hypos/hypo_openai_whisper-large-v2_beam4.*.human_train.txt",
        #"../hypos/hypo_openai_whisper-large-v2_cv_filtered_beam4.*.txt",
        "../hypos/hypo_openai_whisper-large-v2_beam4.numbers_youtube.txt",
        ]

    for files in allfiles:
        for file in glob(files):
            outfile = open(file.replace("large-v2","large-v2_converted"),"w")

            lines = open(file).readlines()
            #random.shuffle(lines)

            for i,line in enumerate(tqdm(lines)):
                #print(i)

                line = line.strip().split()
                id, hypo = line[0], " ".join(line[1:])

                response, _ = run_llm(hypo, number=number, model=model)
                outfile.write(f"{id} {response}\n")

                #if i > 200:
                #    break

    print(f"Price in total: {price:.1f} $ cents")


