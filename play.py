"""CLI for the emotional-poker harness.

Agents are named on the command line. Baselines need no model; an ``llm:...``
agent lazily loads the control-vector Qwen3 and steers it with the given
emotions, e.g.::

    # exploitability table for the baselines (instant, no model)
    python play.py exploit random nash always_call always_fold

    # how badly does a risk-cranked LLM play? (loads the model)
    python play.py exploit "llm:risk=6" "llm:risk=-6" nash

    # head-to-head: euphoric vs cautious LLM, 200 hands
    python play.py match "llm:happiness=6" "llm:risk=-6" --hands 200

Agent spec grammar:
    random | nash | always_call | always_fold
    llm[:vec=strength,vec=strength,...][@samples]   (e.g. llm:risk=6@8)
"""

from __future__ import annotations

import argparse
import sys

from poker import (
    load_game,
    RandomAgent,
    AlwaysCallAgent,
    AlwaysFoldAgent,
    NashAgent,
    LLMAgent,
    head_to_head,
    exploitability_of,
)

BASELINES = {
    "random": RandomAgent,
    "always_call": AlwaysCallAgent,
    "always_fold": AlwaysFoldAgent,
}

# Lazily-populated singletons so the heavy model loads at most once.
_GAME = None
_MODEL_BUNDLE = None  # (model, tokenizer, specs)


def get_game(name: str):
    global _GAME
    if _GAME is None or _GAME.get_type().short_name != name:
        _GAME = load_game(name)
    return _GAME


def get_model_bundle():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        print("Loading control-vector model (first run trains/caches vectors)...",
              file=sys.stderr)
        from vectors import load_model_and_vectors
        _MODEL_BUNDLE = load_model_and_vectors()
        print("Model ready.", file=sys.stderr)
    return _MODEL_BUNDLE


def parse_emotions(body: str) -> dict[str, float]:
    """'risk=6,happiness=-3' -> {'risk': 6.0, 'happiness': -3.0}."""
    emotions = {}
    for part in filter(None, body.split(",")):
        name, _, value = part.partition("=")
        emotions[name.strip()] = float(value)
    return emotions


def make_agent(spec: str, game):
    """Build an agent from a spec string (see module docstring)."""
    spec = spec.strip()
    if spec in BASELINES:
        return BASELINES[spec]()
    if spec == "nash":
        return NashAgent.from_cfr(game)
    if spec == "llm" or spec.startswith("llm:") or spec.startswith("llm@"):
        # llm[:emotions][@samples]
        rest = spec[3:]
        samples = 1
        if "@" in rest:
            rest, _, s = rest.partition("@")
            samples = int(s)
        body = rest[1:] if rest.startswith(":") else ""
        emotions = parse_emotions(body)
        model, tokenizer, specs = get_model_bundle()
        label = "llm[" + (",".join(f"{k}{v:+g}" for k, v in emotions.items()) or "neutral") + "]"
        return LLMAgent(
            model=model, tokenizer=tokenizer, specs=specs,
            emotions=emotions, name=label, samples=samples,
            temperature=0.7 if samples > 1 else 0.0,
        )
    raise SystemExit(f"unknown agent spec: {spec!r}")


def cmd_exploit(args):
    game = get_game(args.game)
    agents = [make_agent(s, game) for s in args.agents]
    print(f"\nExploitability on {args.game} "
          f"(chips/hand a perfect opponent wins; 0 = optimal):\n")
    for a in agents:
        e = exploitability_of(game, a)
        print(f"  {a.name:28s} {e.value:.4f}")
    print()


def cmd_match(args):
    game = get_game(args.game)
    a = make_agent(args.agent_a, game)
    b = make_agent(args.agent_b, game)
    r = head_to_head(game, a, b, hands=args.hands, seed=args.seed)
    print(f"\n{r}\n")
    sign = "wins" if r.mean_return > 0 else "loses" if r.mean_return < 0 else "ties"
    print(f"  {a.name} {sign} {abs(r.mean_return):.4f} chips/hand vs {b.name}.\n")


def main():
    p = argparse.ArgumentParser(description="Emotional-poker harness")
    p.add_argument("--game", default="kuhn_poker",
                   help="OpenSpiel game (kuhn_poker, leduc_poker, ...)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("exploit", help="exact exploitability of one or more agents")
    pe.add_argument("agents", nargs="+", help="agent specs")
    pe.set_defaults(func=cmd_exploit)

    pm = sub.add_parser("match", help="head-to-head between two agents")
    pm.add_argument("agent_a")
    pm.add_argument("agent_b")
    pm.add_argument("--hands", type=int, default=1000)
    pm.add_argument("--seed", type=int, default=0)
    pm.set_defaults(func=cmd_match)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
