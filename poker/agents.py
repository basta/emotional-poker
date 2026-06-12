"""Agents = policies.

Every agent implements one method::

    action_probs(view) -> {action_id: probability}

That is *all* the harness needs. Sampling from it plays a hand; tabulating it
over every information state yields an exact policy whose exploitability we can
compute (see ``poker.eval``). Baseline agents are a few lines each; the LLM agent
prompts a control-vector-steered Qwen3 and parses its chosen verb.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from poker.game import GameView, PASS, BET


class Agent:
    """Base policy interface. Override ``action_probs``."""

    name: str = "agent"

    def action_probs(self, view: GameView) -> dict[int, float]:
        raise NotImplementedError

    def act(self, view: GameView, rng) -> int:
        """Sample a concrete action from the agent's distribution."""
        probs = self.action_probs(view)
        actions = list(probs)
        weights = [probs[a] for a in actions]
        return rng.choices(actions, weights=weights, k=1)[0]


# --------------------------------------------------------------------------- #
# Baselines -- reference anchors for the eval.
# --------------------------------------------------------------------------- #


class RandomAgent(Agent):
    """Uniform over legal actions. Exploitability anchor (~0.458 in Kuhn)."""

    name = "random"

    def action_probs(self, view: GameView) -> dict[int, float]:
        p = 1.0 / len(view.legal_actions)
        return {a: p for a in view.legal_actions}


class AlwaysCallAgent(Agent):
    """Never folds, never bets: the 'calling station'. Always picks BET=call,
    and checks when nothing is live (i.e. never folds, never raises)."""

    name = "always_call"

    def action_probs(self, view: GameView) -> dict[int, float]:
        # Facing a bet -> call (BET). Otherwise check (PASS).
        choice = BET if view.facing_bet else PASS
        return {a: (1.0 if a == choice else 0.0) for a in view.legal_actions}


class AlwaysFoldAgent(Agent):
    """The 'nit' that folds to any bet and never bets itself (always PASS)."""

    name = "always_fold"

    def action_probs(self, view: GameView) -> dict[int, float]:
        return {a: (1.0 if a == PASS else 0.0) for a in view.legal_actions}


@dataclass
class NashAgent(Agent):
    """Wraps a solved OpenSpiel policy (e.g. CFR average) as an agent.

    This is the *optimal opponent*: near-zero exploitability. Build it with
    ``NashAgent.from_cfr(game)``.
    """

    spiel_policy: object  # open_spiel.python.policy.Policy
    name: str = "nash"
    # info_state string -> {action: prob}, filled lazily from the policy.
    _table: dict = field(default_factory=dict)

    @classmethod
    def from_cfr(cls, game, iterations: int = 2000) -> "NashAgent":
        from open_spiel.python.algorithms import cfr

        solver = cfr.CFRSolver(game)
        for _ in range(iterations):
            solver.evaluate_and_update_policy()
        return cls(spiel_policy=solver.average_policy())

    def action_probs(self, view: GameView) -> dict[int, float]:
        # The wrapped policy is keyed by info-state string via a state object;
        # we cache by string so we never need a live state here.
        if view.info_state not in self._table:
            self._table[view.info_state] = self._lookup(view)
        return self._table[view.info_state]

    def _lookup(self, view: GameView) -> dict[int, float]:
        # The spiel policy exposes probabilities per state; the eval module
        # pre-populates the table from real states. As a fallback, default to
        # uniform (only hit if queried for an unseen info state).
        p = 1.0 / len(view.legal_actions)
        return {a: p for a in view.legal_actions}

    def prime_from_states(self, states) -> None:
        """Fill the lookup table from concrete OpenSpiel states (exact)."""
        for s in states:
            probs = self.spiel_policy.action_probabilities(s)
            info = s.information_state_string()
            self._table[info] = {int(a): float(p) for a, p in probs.items()}


# --------------------------------------------------------------------------- #
# LLM agent -- the thing we actually want to measure.
# --------------------------------------------------------------------------- #

ACTION_RE = re.compile(r"\b(check|bet|call|fold)\b", re.IGNORECASE)


@dataclass
class LLMAgent(Agent):
    """A control-vector-steered Qwen3 playing poker via natural language.

    Holds a reference to the shared (model, tokenizer, specs) from ``vectors.py``
    and an ``emotions`` dict {vector_name: strength}. Before each decision it
    injects the blended control vector, prompts the model with the situation,
    and parses the chosen verb. ``samples > 1`` queries the model repeatedly at
    nonzero temperature to estimate a real action distribution (cheap: Kuhn has
    only 12 info states).
    """

    model: object
    tokenizer: object
    specs: list  # list[VectorSpec] from vectors.py
    emotions: dict = field(default_factory=dict)  # name -> strength
    name: str = "llm"
    samples: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 256
    think: bool = False  # let Qwen3 emit <think>; slower but emotion acts on it
    verbose: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        # Pre-blend the control vector once from the emotion strengths.
        self._control = None
        by_name = {s.name: s for s in self.specs}
        for vname, strength in self.emotions.items():
            if not strength:
                continue
            spec = by_name.get(vname)
            if spec is None or spec.vector is None:
                raise KeyError(f"unknown/untrained emotion vector: {vname!r}")
            term = spec.vector * float(strength)
            self._control = term if self._control is None else self._control + term

    # --- prompting -------------------------------------------------------- #

    def _prompt(self, view: GameView) -> str:
        verbs = " or ".join(view.verbs[a] for a in view.legal_actions)
        think_tag = "" if self.think else " /no_think"
        return (
            "We are playing Kuhn poker. There are three cards total — Jack, "
            "Queen, King (King is highest). You and your opponent are each dealt "
            "one distinct card; the third is unused, so you each hold a different "
            "card. Both of us ante 1 chip. On your turn you may check or bet 1 "
            "chip; facing a bet you may call or fold. At showdown the higher card "
            "wins the pot.\n\n"
            f"Your card: {view.card_name}.\n"
            f"{view.situation()} {view.action_menu()}\n\n"
            "Reason briefly about the best play, then end with exactly "
            f"'ACTION: <{verbs}>'."
            f"{think_tag}"
        )

    def _generate(self, prompt: str) -> str:
        from vectors import chatml

        inputs = self.tokenizer(chatml(prompt), return_tensors="pt").to(
            self.model.device
        )
        kwargs = dict(
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=1.1,
        )
        if self.temperature and self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with self._lock:
            self.model.reset()
            if self._control is not None:
                self.model.set_control(self._control, 1.0)
            out = self.model.generate(**inputs, **kwargs)
            self.model.reset()
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return text

    def _parse_action(self, text: str, view: GameView) -> int | None:
        """Map the committed verb to a legal action id, or None if undecided.

        With thinking on we only look *after* ``</think>``: a verb mentioned mid
        reasoning is not a decision. If thinking never closed (truncated by the
        token budget) there is no committed action, so we return None and let the
        caller count it as a miss rather than reading tea leaves from the ramble.
        """
        verb_to_action = {v: a for a, v in view.verbs.items()}
        search = text
        if self.think:
            if "</think>" not in text:
                return None  # never finished reasoning -> no committed move
            search = text.rsplit("</think>", 1)[1]
        # Prefer an explicit 'ACTION: x' tail; else the last verb in the answer.
        tail = search.split("ACTION:")[-1] if "ACTION:" in search else search
        matches = ACTION_RE.findall(tail) or ACTION_RE.findall(search)
        for verb in reversed(matches):
            a = verb_to_action.get(verb.lower())
            if a is not None:
                return a
        return None

    def action_probs(self, view: GameView) -> dict[int, float]:
        counts = {a: 0 for a in view.legal_actions}
        n = max(1, self.samples)
        misses = 0
        for _ in range(n):
            text = self._generate(self._prompt(view))
            a = self._parse_action(text, view)
            if self.verbose:
                print(f"[{self.name} {view.info_state}] -> {a}: {text[:120]!r}")
            if a is None:
                misses += 1
                continue
            counts[a] += 1
        total = sum(counts.values())
        if total == 0:
            # Model never produced a parseable action: fall back to uniform so
            # the policy is still well-defined (and flag it).
            if self.verbose:
                print(f"[{self.name} {view.info_state}] no parseable action; uniform")
            p = 1.0 / len(view.legal_actions)
            return {a: p for a in view.legal_actions}
        return {a: c / total for a, c in counts.items()}
