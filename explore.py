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
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
model = ControlModel(model, list(range(-5, -18, -1)))

user_tag = "<|im_start|>user\n"
asst_tag = "<|im_end|>\n<|im_start|>assistant\n"

with open("true_facts.json") as f:
    suffixes = json.load(f)

# the control vector we're going to make is honest / untruthful, like the paper
positive_personas = ["honest"]
negative_personas = ["untruthful"]


def template(persona: str, suffix: str) -> str:
    return f"{user_tag} Pretend you're an {persona} person making statements about the world. {asst_tag} {suffix}"


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

for i in range(3):
    print(f"dataset[{i}].positive:", dataset[i].positive)
    print(f"dataset[{i}].negative:", dataset[i].negative)


model.reset()  # make sure you always reset the model before training a new vector
control_vector = cached(
    ("control_vector", model_name, positive_personas, negative_personas, len(dataset)),
    lambda: ControlVector.train(model, tokenizer, dataset),
)

# this question is taken from the paper
input = f"{user_tag} You are late for work because party until very late last night, but you don't want to lose your job. What would you tell your boss instead? {asst_tag}"

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

print("\n++control")
# add the control vector with a certain strength (try increasing or decreasing this!)
model.set_control(control_vector, 2)
print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))

print("\n--control")
# subtract the control vector, giving the opposite result (e.g. sad instead of happy)
# depending on your vector, you may need more or less negative strength to match the positive effect
model.set_control(control_vector, -5)
print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()))
model.reset()
