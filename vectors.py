"""Shared model + control-vector setup.

Factors the duplicated logic out of explore.py / explore_risk.py into one place
so the model is loaded once and every vector reuses the same on-disk cache.

The set of vectors is described declaratively in ``vectors_config.json`` — add a
new entry there (and a suffix .json of seed statements) to introduce a vector.
Run ``build_vectors.py`` to train/cache everything ahead of time.
"""

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import torch
from repeng import ControlModel, ControlVector, DatasetEntry
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
CONFIG_FILE = Path("vectors_config.json")

USER_TAG = "<|im_start|>user\n"
ASST_TAG = "<|im_end|>\n<|im_start|>assistant\n"

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_CONTROL_LAYERS = list(range(-5, -18, -1))
DEFAULT_PROMPT_TEMPLATE = (
    "{user}Pretend you're {article} {persona} person.{asst}{suffix}"
)


def _cache_path(key: tuple) -> Path:
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.pkl"


def cached(key: tuple, fn):
    path = _cache_path(key)
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    value = fn()
    with path.open("wb") as f:
        pickle.dump(value, f)
    return value


@dataclass
class VectorSpec:
    """Everything needed to train one named control vector."""

    name: str  # slider label, e.g. "honest"
    positive_persona: str  # high end of the slider
    negative_persona: str  # low end of the slider
    suffix_file: str  # json list of seed statements
    prompt_template: str  # uses {user} {asst} {article} {persona} {suffix}
    suggested_range: tuple = (-10.0, 10.0)
    vector: ControlVector = field(default=None, repr=False)


@dataclass
class Config:
    """Parsed vectors_config.json."""

    model_name: str
    control_layers: list[int]
    specs: list[VectorSpec]


def _parse_control_layers(value) -> list[int]:
    """Accept either an explicit list or a {start, stop, step} range spec."""
    if isinstance(value, dict):
        return list(range(value["start"], value["stop"], value.get("step", -1)))
    return list(value)


def load_config(path: Path = CONFIG_FILE) -> Config:
    """Read the declarative vector config into typed specs."""
    with open(path) as f:
        raw = json.load(f)

    default_template = raw.get("default_prompt_template", DEFAULT_PROMPT_TEMPLATE)
    specs = [
        VectorSpec(
            name=entry["name"],
            positive_persona=entry["positive_persona"],
            negative_persona=entry["negative_persona"],
            suffix_file=entry["suffix_file"],
            prompt_template=entry.get("prompt_template", default_template),
            suggested_range=tuple(entry.get("suggested_range", (-10.0, 10.0))),
        )
        for entry in raw["vectors"]
    ]
    return Config(
        model_name=raw.get("model_name", DEFAULT_MODEL_NAME),
        control_layers=_parse_control_layers(
            raw.get("control_layers", DEFAULT_CONTROL_LAYERS)
        ),
        specs=specs,
    )


def _build_dataset(tokenizer, spec: VectorSpec) -> list[DatasetEntry]:
    with open(spec.suffix_file) as f:
        suffixes = json.load(f)

    def render(persona: str, suffix: str) -> str:
        article = "an" if persona[0] in "aeiou" else "a"
        return spec.prompt_template.format(
            user=USER_TAG, asst=ASST_TAG, article=article, persona=persona, suffix=suffix
        )

    dataset = []
    for suffix in suffixes:
        tokens = tokenizer.tokenize(suffix)
        # augment the short list by training on many truncations of each statement
        for i in range(1, len(tokens) - 5):
            truncated = tokenizer.convert_tokens_to_string(tokens[:i])
            dataset.append(
                DatasetEntry(
                    positive=render(spec.positive_persona, truncated),
                    negative=render(spec.negative_persona, truncated),
                )
            )
    return dataset


def _vector_cache_key(model_name: str, tokenizer, spec: VectorSpec) -> tuple:
    """Cache key for a spec: model + the exact training pairs it would produce."""
    dataset = _build_dataset(tokenizer, spec)
    fingerprint = hashlib.sha256(
        repr([(d.positive, d.negative) for d in dataset]).encode()
    ).hexdigest()
    return ("control_vector", model_name, fingerprint), dataset


def is_cached(model_name: str, tokenizer, spec: VectorSpec) -> bool:
    """Whether this spec's vector is already trained on disk."""
    key, _ = _vector_cache_key(model_name, tokenizer, spec)
    return _cache_path(key).exists()


def train_vector(model, tokenizer, spec: VectorSpec, model_name: str) -> bool:
    """Train (or load from cache) one spec's vector, storing it on ``spec.vector``.

    Returns True if it was loaded from cache, False if freshly trained.
    """
    model.reset()  # always reset before training a new vector
    key, dataset = _vector_cache_key(model_name, tokenizer, spec)
    was_cached = _cache_path(key).exists()
    spec.vector = cached(key, lambda: ControlVector.train(model, tokenizer, dataset))
    return was_cached


def build_model(config: Config):
    """Load the base model and wrap it as a ControlModel per the config."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.float16
    )
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ControlModel(model, config.control_layers)
    return model, tokenizer


def load_model_and_vectors(config_path: Path = CONFIG_FILE):
    """Load the model once and train/cache every configured vector.

    Returns (model, tokenizer, specs) with spec.vector populated.
    """
    config = load_config(config_path)
    model, tokenizer = build_model(config)

    for spec in config.specs:
        train_vector(model, tokenizer, spec, config.model_name)

    model.reset()
    return model, tokenizer, config.specs


def chatml(message: str) -> str:
    """Wrap a user message in the model's chat template."""
    return f"{USER_TAG}{message}{ASST_TAG}"
