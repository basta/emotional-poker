"""Emotional-poker harness.

A tiny, evaluation-focused poker harness built on OpenSpiel. Agents are nothing
but *policies* (info-state -> action distribution); that single interface drives
both head-to-head play and exact exploitability measurement.

Start with Kuhn poker (12 information states, closed-form optimum) and step up to
Leduc later by changing one game string -- see ``poker.game.load_game``.
"""

from poker.game import GameView, load_game
from poker.agents import (
    Agent,
    RandomAgent,
    AlwaysCallAgent,
    AlwaysFoldAgent,
    NashAgent,
    LLMAgent,
)
from poker.eval import (
    head_to_head,
    exploitability_of,
    agent_to_policy,
    exact_value,
    exact_value_of_policies,
)

__all__ = [
    "GameView",
    "load_game",
    "Agent",
    "RandomAgent",
    "AlwaysCallAgent",
    "AlwaysFoldAgent",
    "NashAgent",
    "LLMAgent",
    "head_to_head",
    "exploitability_of",
    "agent_to_policy",
    "exact_value",
    "exact_value_of_policies",
]
