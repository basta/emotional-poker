"""Game loading + human-readable rendering of OpenSpiel poker states.

Everything an agent (or a prompted LLM) needs to reason about a decision is
derived here from a single OpenSpiel ``state``: which card it holds, what has
happened so far, and what the two legal actions *mean* in context (check/bet vs
fold/call). Kuhn poker is the default; ``load_game("leduc_poker")`` reuses the
same machinery once we step up.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyspiel

# Kuhn deals one of three cards. OpenSpiel encodes them as 0/1/2.
KUHN_CARD_NAMES = {0: "Jack", 1: "Queen", 2: "King"}

# Kuhn / Leduc share these two primitive actions. Action 0 is the "passive"
# move (check when no bet is live, fold when one is); action 1 is the
# "aggressive" move (bet when none is live, call when one is).
PASS, BET = 0, 1


def load_game(name: str = "kuhn_poker") -> pyspiel.Game:
    """Load an OpenSpiel game. Default Kuhn; ``leduc_poker`` is the step-up."""
    return pyspiel.load_game(name)


def _kuhn_history_letters(info_state: str) -> str:
    """Strip the leading card digit off a Kuhn info-state string.

    Info-state strings look like ``"0"``, ``"0pb"``, ``"2p"`` -- a card digit
    followed by a sequence of ``p`` (pass) / ``b`` (bet) letters.
    """
    return info_state[1:]


def facing_bet(history_letters: str) -> bool:
    """True if a bet is live in front of the player (last action was a bet)."""
    return history_letters.endswith("b")


def action_verbs(history_letters: str) -> dict[int, str]:
    """Map the two legal actions to their contextual verbs."""
    if facing_bet(history_letters):
        return {PASS: "fold", BET: "call"}
    return {PASS: "check", BET: "bet"}


@dataclass
class GameView:
    """A flattened, render-ready snapshot of one decision point.

    Built from an OpenSpiel state at a player node. ``card``/``card_name`` are the
    acting player's private card; ``verbs`` translates each legal action id into
    the word a human (or LLM) would use for it.
    """

    player: int
    card: int
    card_name: str
    info_state: str
    history_letters: str
    legal_actions: list[int]
    verbs: dict[int, str]
    facing_bet: bool

    @classmethod
    def from_state(cls, state: pyspiel.State) -> "GameView":
        info_state = state.information_state_string()
        card = int(info_state[0])
        history = _kuhn_history_letters(info_state)
        return cls(
            player=state.current_player(),
            card=card,
            card_name=KUHN_CARD_NAMES.get(card, str(card)),
            info_state=info_state,
            history_letters=history,
            legal_actions=list(state.legal_actions()),
            verbs=action_verbs(history),
            facing_bet=facing_bet(history),
        )

    def situation(self) -> str:
        """One-line description of what just happened, for prompting."""
        h = self.history_letters
        if h == "":
            return "You are first to act."
        if h == "p":
            return "Your opponent checked to you."
        if h == "b":
            return "Your opponent bet."
        if h == "pb":
            return "You checked; your opponent then bet."
        return f"History so far: {h}."

    def action_menu(self) -> str:
        """e.g. 'You can either check (stay in for free) or bet (put 1 in).'"""
        opts = " or ".join(
            f"{self.verbs[a]}" for a in self.legal_actions
        )
        return f"You can {opts}."
