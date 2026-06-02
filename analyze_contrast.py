"""
Which statements actually drive the risk-taking control vector?

For every (statement, truncation) pair we run the positive (risk-taking) and
negative (cautious) prompts through the model, read the last-token hidden state
at each controlled layer, and take the difference d = h_pos - h_neg.

We then project d onto the *learned* control-vector direction for that layer.
That projection IS the per-example contribution to the contrast the PCA picked
up. We standardize per layer (so layers with different activation scales are
comparable), average across layers and across a statement's truncations, and
rank statements by the result.
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from repeng import ControlModel, ControlVector, DatasetEntry
from repeng.control import model_layer_list
from repeng.extract import batched_get_hiddens, project_onto_direction
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
use_cuda = torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"
dtype = torch.float16 if use_cuda else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name)
base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
control_layers = list(range(-5, -18, -1))
model = ControlModel(base, control_layers)

user_tag = "<|im_start|>user\n"
asst_tag = "<|im_end|>\n<|im_start|>assistant\n"

with open("risk_statements.json") as f:
    suffixes = json.load(f)

positive_personas = ["risk-taking"]
negative_personas = ["cautious"]


def template(persona: str, suffix: str) -> str:
    article = "an" if persona[0] in "aeiou" else "a"
    return f"{user_tag}Pretend you're {article} {persona} person making decisions about what to do.{asst_tag}{suffix}"


# Build the dataset exactly as explore_risk.py does, but remember which
# statement each entry came from.
dataset = []
entry_stmt = []  # statement index for each dataset entry
for si, suffix in enumerate(suffixes):
    tokens = tokenizer.tokenize(suffix)
    for i in range(1, len(tokens) - 5):
        truncated = tokenizer.convert_tokens_to_string(tokens[:i])
        for p, n in zip(positive_personas, negative_personas):
            dataset.append(
                DatasetEntry(positive=template(p, truncated), negative=template(n, truncated))
            )
            entry_stmt.append(si)

entry_stmt = np.array(entry_stmt)

# Reuse the exact same cached control vector explore_risk.py trained.
model.reset()
dataset_fingerprint = hashlib.sha256(
    repr([(d.positive, d.negative) for d in dataset]).encode()
).hexdigest()
control_vector = cached(
    ("control_vector", model_name, dataset_fingerprint),
    lambda: ControlVector.train(model, tokenizer, dataset),
)

# Read last-token hidden states for every prompt at the controlled layers.
n_layers = len(model_layer_list(model))
hidden_layers = [n_layers + i for i in control_layers]  # normalize to positive ids
train_strs = [s for ex in dataset for s in (ex.positive, ex.negative)]

model.reset()
print(f"running {len(train_strs)} forward passes on {device} ...")
H = batched_get_hiddens(model, tokenizer, train_strs, hidden_layers, batch_size=32)

# Per layer: diff = pos - neg, project onto the learned direction, z-score across
# all examples so layers are comparable.
N = len(dataset)
z_per_layer = []
mag_per_layer = []
for layer in hidden_layers:
    h = H[layer]  # (2N, d), order [pos, neg, pos, neg, ...]
    diff = h[0::2] - h[1::2]  # (N, d)
    proj = project_onto_direction(diff, control_vector.directions[layer])  # (N,)
    z = (proj - proj.mean()) / (proj.std() + 1e-8)
    z_per_layer.append(z)
    mag_per_layer.append(np.linalg.norm(diff, axis=1))

entry_score = np.mean(z_per_layer, axis=0)  # (N,) standardized alignment with the axis
entry_mag = np.mean(mag_per_layer, axis=0)  # (N,) raw contrast magnitude

# Aggregate per statement.
stmt_score = np.array([entry_score[entry_stmt == s].mean() for s in range(len(suffixes))])
stmt_mag = np.array([entry_mag[entry_stmt == s].mean() for s in range(len(suffixes))])
stmt_n = np.array([(entry_stmt == s).sum() for s in range(len(suffixes))])

# Groups (by line ranges in the file).
groups = {
    "risk-endorsing (idx 0-39)": range(0, 40),
    "caution-endorsing (idx 40-78)": range(40, 79),
    "neutral filler (idx 79-98)": range(79, len(suffixes)),
}

print("\n================ GROUP SUMMARY ================")
print(f"{'group':36s} {'n':>3s} {'mean_align':>11s} {'mean_mag':>9s}")
for name, rng in groups.items():
    idx = [i for i in rng if i < len(suffixes)]
    print(f"{name:36s} {len(idx):3d} {stmt_score[idx].mean():11.3f} {stmt_mag[idx].mean():9.2f}")

order = np.argsort(-stmt_score)


def show(title, idxs):
    print(f"\n================ {title} ================")
    print(f"{'align':>7s} {'mag':>7s} {'trunc':>5s}  statement")
    for i in idxs:
        s = suffixes[i][:78]
        print(f"{stmt_score[i]:7.3f} {stmt_mag[i]:7.2f} {stmt_n[i]:5d}  [{i:2d}] {s}")


show("TOP 15 CONTRAST DRIVERS", order[:15])
show("BOTTOM 15 (weak / opposing)", order[-15:])

# Save full ranking for later use.
ranking = sorted(
    ({"idx": int(i), "align": float(stmt_score[i]), "mag": float(stmt_mag[i]),
      "n_trunc": int(stmt_n[i]), "statement": suffixes[i]} for i in range(len(suffixes))),
    key=lambda r: -r["align"],
)
with open("contrast_ranking.json", "w") as f:
    json.dump(ranking, f, indent=2)
print("\nfull ranking written to contrast_ranking.json")
