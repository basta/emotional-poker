import hashlib
import json
import pickle
from pathlib import Path

import torch
from repeng import ControlModel, ControlVector, DatasetEntry
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def cached(key: tuple, fn):
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{digest}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    value = fn()
    with path.open("wb") as f:
        pickle.dump(value, f)
    return value


model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
model = ControlModel(model, list(range(-5, -18, -1)))

user_tag = "<|im_start|>user\n"
asst_tag = "<|im_end|>\n<|im_start|>assistant\n"

with open("risk_statements.json") as f:
    suffixes = json.load(f)

# the control vector we're going to make is risk-taking / cautious
positive_personas = ["risk-taking"]
negative_personas = ["cautious"]


def template(persona: str, suffix: str) -> str:
    article = "an" if persona[0] in "aeiou" else "a"
    return f"{user_tag}Pretend you're {article} {persona} person making decisions about what to do.{asst_tag}{suffix}"


dataset = []
for suffix in suffixes:
    tokens = tokenizer.tokenize(suffix)
    # we augment our short suffix list by taking lots of different truncations.
    # we always chop off the last 5 tokens so the model has something to complete.
    for i in range(1, len(tokens) - 5):
        truncated = tokenizer.convert_tokens_to_string(tokens[:i])
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):
            dataset.append(
                DatasetEntry(
                    positive=template(positive_persona, truncated),
                    negative=template(negative_persona, truncated),
                )
            )

model.reset()  # make sure you always reset the model before training a new vector
dataset_fingerprint = hashlib.sha256(
    repr([(d.positive, d.negative) for d in dataset]).encode()
).hexdigest()
control_vector = cached(
    ("control_vector", model_name, dataset_fingerprint),
    lambda: ControlVector.train(model, tokenizer, dataset),
)

# a decision the model has to make where risk attitude matters
input = f"{user_tag}I have $10,000 in savings. Should I invest it all in a risky new startup, or keep it in the bank? Tell me what to do. A single sentence is enough{asst_tag}"

# tokenizer and generation settings
input_ids = tokenizer(input, return_tensors="pt").to(model.device)
settings = {
    "pad_token_id": tokenizer.eos_token_id,  # silence warning
    "do_sample": False,  # temperature=0
    "max_new_tokens": 128,
    "repetition_penalty": 1.1,  # reduce control jank
}

print("==baseline")
model.reset()
print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))

# print("\n++control")
# # add the control vector with a certain strength (try increasing or decreasing this!)
# model.set_control(control_vector, 2)
# print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))

for i in range(-10, 10, 2):
    print(f"\n--control: {i}")
    # subtract the control vector, giving the opposite result (e.g. cautious instead of risk-taking)
    # depending on your vector, you may need more or less negative strength to match the positive effect
    model.set_control(control_vector, i)
    print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))
    model.reset()
