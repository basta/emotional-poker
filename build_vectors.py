"""Pre-build and cache every control vector described in vectors_config.json.

Loads the model once, then for each configured vector checks the on-disk cache
and only trains the ones that are missing. Run this after adding a new entry to
``vectors_config.json`` so the app starts up instantly:

    uv run build_vectors.py
    uv run build_vectors.py --config other_config.json
    uv run build_vectors.py --force        # retrain even if cached
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from vectors import CONFIG_FILE, build_model, is_cached, load_config, train_vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIG_FILE, help="path to vectors config json"
    )
    parser.add_argument(
        "--force", action="store_true", help="retrain vectors even if already cached"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config}  ({len(config.specs)} vectors)")
    print(f"Model:  {config.model_name}")
    print(f"Layers: {config.control_layers}\n")

    # The cache key depends on tokenization, so we need the tokenizer to tell
    # what's missing — but it's cheap next to the model weights.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    missing = [
        spec
        for spec in config.specs
        if args.force or not is_cached(config.model_name, tokenizer, spec)
    ]
    for spec in config.specs:
        if spec not in missing:
            print(f"  [cached] {spec.name}  ({spec.suffix_file})")

    if not missing:
        print("\nAll vectors already cached. Nothing to do.")
        return

    print(f"\nLoading model to train {len(missing)} vector(s)...")
    model, _ = build_model(config)

    for spec in missing:
        print(f"  [train ] {spec.name}  ({spec.suffix_file}) ...", flush=True)
        train_vector(model, tokenizer, spec, config.model_name)
        print(f"  [done  ] {spec.name}")

    print("\nAll vectors are cached and ready.")


if __name__ == "__main__":
    main()
