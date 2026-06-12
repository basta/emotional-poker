"""Sweep one emotion vector across strengths and watch poker skill degrade.

Loads the control-vector model once, then for each strength builds an
LLMAgent steered by that single emotion and measures its exact exploitability
(how badly it plays) plus its head-to-head result against the Nash optimum.

    python sweep.py risk --strengths -8 -4 0 4 8
    python sweep.py happiness --strengths -6 0 6 --hands 200 --samples 4

The neutral (strength 0) row is your baseline; a rising exploitability column is
the emotion making the model play worse.
"""

from __future__ import annotations

import argparse

from open_spiel.python.algorithms import exploitability as exploit_alg

from poker import (
    load_game,
    LLMAgent,
    NashAgent,
    agent_to_policy,
    exact_value_of_policies,
)
from vectors import load_model_and_vectors


def main():
    p = argparse.ArgumentParser(description="Emotion -> poker-skill sweep")
    p.add_argument("emotion", help="vector name (e.g. risk, happiness, honesty)")
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[-8, -4, 0, 4, 8])
    p.add_argument("--game", default="kuhn_poker")
    p.add_argument("--samples", type=int, default=1,
                   help=">1 estimates a stochastic policy by repeated sampling")
    p.add_argument("--think", action="store_true",
                   help="let Qwen3 reason in <think>...</think> (emotion steers "
                        "the reasoning). Implies sampling; bump --samples for a "
                        "stable distribution.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="token budget per decision (default 1024 with --think, "
                        "else 256)")
    p.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature (default 0.6 with --think, 0.7 "
                        "when --samples>1, else 0 greedy)")
    args = p.parse_args()

    max_new_tokens = args.max_new_tokens or (1024 if args.think else 256)
    if args.temperature is not None:
        temperature = args.temperature
    elif args.think:
        temperature = 0.6          # Qwen3's recommended thinking temperature
    elif args.samples > 1:
        temperature = 0.7
    else:
        temperature = 0.0          # greedy

    game = load_game(args.game)
    model, tokenizer, specs = load_model_and_vectors()
    if args.emotion not in {s.name for s in specs}:
        raise SystemExit(
            f"unknown emotion {args.emotion!r}; available: "
            f"{', '.join(s.name for s in specs)}"
        )
    nash = NashAgent.from_cfr(game)
    nash_policy = agent_to_policy(game, nash)  # primes + tabulates once

    print(f"\nSweeping '{args.emotion}' on {args.game} "
          f"(think={args.think}, samples={args.samples}, temp={temperature})")
    print("exploitability = chips/hand a perfect opponent wins (0 = optimal); "
          "vs Nash is exact.\n")
    print(f"  {'strength':>9}  {'exploitability':>14}  {'vs Nash (chips/hand)':>22}")
    print("  " + "-" * 49)
    for strength in args.strengths:
        agent = LLMAgent(
            model=model, tokenizer=tokenizer, specs=specs,
            emotions={args.emotion: strength},
            name=f"llm[{args.emotion}{strength:+g}]",
            samples=args.samples,
            temperature=temperature,
            think=args.think,
            max_new_tokens=max_new_tokens,
        )
        # Tabulate the LLM policy once (the only generations), derive both metrics.
        policy = agent_to_policy(game, agent)
        exploit = exploit_alg.exploitability(game, policy)
        vs_nash = exact_value_of_policies(game, policy, nash_policy)
        flag = "  <- baseline" if strength == 0 else ""
        print(f"  {strength:>+9g}  {exploit:>14.4f}  {vs_nash:>+22.4f}{flag}",
              flush=True)
    print()


if __name__ == "__main__":
    main()
