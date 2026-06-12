# emotional-poker

Steer Qwen3-4B with emotion **control vectors** (risk, happiness, honesty, …) and
measure how that emotion degrades its *poker* decisions against a computable
optimum.

The harness uses **Kuhn poker** (3 cards, one bet round) via
[OpenSpiel](https://github.com/google-deepmind/open_spiel). Kuhn has only **12
information states** and a closed-form Nash equilibrium, so we don't just measure
noisy win-rates — we enumerate every info state, read the agent's full policy,
and compute its **exact exploitability**: the chips per hand a perfect opponent
wins against it. `0` = optimal; the `random` baseline is `0.458`. Stepping up to
Leduc later is a one-line game change (`--game leduc_poker`).

## Concepts

- **An agent is a policy**: `action_probs(view) -> {action: prob}`. Sampling it
  plays a hand; tabulating it over all info states gives an exact policy whose
  exploitability we can compute. That's the whole interface (`poker/agents.py`).
- **Baselines** (`poker/agents.py`): `random`, `always_call`, `always_fold`,
  `nash` (CFR-solved optimum). Reference anchors for the eval.
- **LLM agent**: prompts the control-vector Qwen3 with the situation in natural
  language, injects the blended emotion vector, parses the chosen verb.

## Usage

```bash
# Exploitability of the baselines (instant, no model load)
python play.py exploit random nash always_call always_fold

# How badly does the (neutral) LLM play? Loads the model; 12 generations.
python play.py exploit llm

# Steered LLM vs steered LLM, head-to-head
python play.py match "llm:risk=6" "llm:risk=-6" --hands 200

# The core experiment: sweep one emotion, watch skill degrade
python sweep.py risk --strengths -8 -4 0 4 8
python sweep.py happiness --strengths -6 0 6 --samples 4
```

### Agent spec grammar (for `play.py`)

```
random | nash | always_call | always_fold
llm[:vec=strength,vec=strength,...][@samples]      e.g.  llm:risk=6,happiness=-3@8
```

`@samples > 1` queries the model repeatedly at temperature 0.7 to estimate a
*stochastic* policy (mixed strategies matter in poker); `@1` (default) is greedy.

## Layout

| File | Role |
|------|------|
| `poker/game.py`   | load game; render a state into a `GameView` (card, situation, legal verbs) |
| `poker/agents.py` | `Agent` interface + baselines + `LLMAgent` |
| `poker/eval.py`   | `head_to_head` (chip EV) and `exploitability_of` (exact, variance-free) |
| `play.py`         | CLI: `exploit` and `match` |
| `sweep.py`        | sweep one emotion across strengths → exploitability curve |

The control-vector model setup lives in `vectors.py` / `vectors_config.json`
(shared with `app.py`, the single-prompt steering lab).
