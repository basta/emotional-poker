"""Two ways to measure an agent.

1. ``head_to_head`` -- sit two agents at the table for N hands (seats swapped
   every other hand to cancel positional advantage) and report chip EV. This is
   the noisy, "who wins" view.

2. ``exploitability_of`` -- turn one agent into a full tabular policy by querying
   it at *every* information state, then compute its exact exploitability: how
   many chips a perfect best-responder wins against it per hand. Zero = unbeatable
   (Nash). This is the variance-free "how badly does it play" number, and it is
   the headline metric for the emotion experiment.

Both work unchanged for Leduc once you swap the game string -- exploitability
just enumerates more info states.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability, expected_game_score

from poker.agents import Agent, NashAgent
from poker.game import GameView


def _decision_states(game) -> list:
    """Every reachable player-decision state (one per info state in Kuhn)."""
    out = []
    seen = set()

    def walk(s):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        info = s.information_state_string()
        if info not in seen:
            seen.add(info)
            out.append(s.clone())
        for a in s.legal_actions():
            walk(s.child(a))

    walk(game.new_initial_state())
    return out


def agent_to_policy(game, agent: Agent) -> policy_lib.TabularPolicy:
    """Materialize an agent as an exact OpenSpiel tabular policy.

    Queries the agent once per information state. For the LLM agent this is where
    the (few) generations happen -- 12 calls for Kuhn.
    """
    # NashAgent needs real states to read its wrapped policy; prime it first.
    if isinstance(agent, NashAgent):
        agent.prime_from_states(_decision_states(game))

    tab = policy_lib.TabularPolicy(game)
    for state in tab.states:
        view = GameView.from_state(state)
        probs = agent.action_probs(view)
        row = tab.action_probability_array[tab.state_index(state)]
        row[:] = 0.0
        for action, p in probs.items():
            row[action] = p
        # Guard against an all-zero row (e.g. agent returned nothing legal).
        if row.sum() == 0:
            row[view.legal_actions] = 1.0 / len(view.legal_actions)
        else:
            row /= row.sum()
    return tab


@dataclass
class Exploitability:
    value: float          # chips/hand a best-responder wins (averaged over seats)
    policy: policy_lib.TabularPolicy

    def __repr__(self):
        return f"Exploitability(value={self.value:.4f})"


def exploitability_of(game, agent: Agent) -> Exploitability:
    """Exact exploitability of an agent. Lower is better; ~0 is optimal."""
    tab = agent_to_policy(game, agent)
    value = exploitability.exploitability(game, tab)
    return Exploitability(value=value, policy=tab)


def exact_value_of_policies(game, policy_a, policy_b) -> float:
    """Exact, seat-averaged chips/hand for policy_a vs policy_b (no sampling).

    Reuses already-tabulated policies, so an LLM agent is only queried the 12
    times it took to build its policy -- not once per simulated hand.
    """
    init = game.new_initial_state()
    a_as_p0 = expected_game_score.policy_value(init, [policy_a, policy_b])[0]
    a_as_p1 = expected_game_score.policy_value(init, [policy_b, policy_a])[1]
    return 0.5 * (a_as_p0 + a_as_p1)


def exact_value(game, agent_a: Agent, agent_b: Agent) -> float:
    """Exact, seat-averaged chips/hand for agent_a vs agent_b.

    The deterministic counterpart to ``head_to_head`` -- correct for any agents
    whose policy is stable across queries (all baselines; the LLM at @1/greedy).
    """
    return exact_value_of_policies(
        game, agent_to_policy(game, agent_a), agent_to_policy(game, agent_b)
    )


@dataclass
class MatchResult:
    hands: int
    agent0: str
    agent1: str
    # Mean chips won per hand from agent0's perspective (seat-averaged).
    mean_return: float
    stderr: float

    def __repr__(self):
        return (
            f"MatchResult({self.agent0} vs {self.agent1}: "
            f"{self.mean_return:+.4f} ± {self.stderr:.4f} chips/hand "
            f"over {self.hands} hands)"
        )


def _play_one(game, state, agents, rng) -> float:
    """Play a dealt hand to terminal; return player 0's payoff."""
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(rng.choices(outcomes, weights=probs, k=1)[0])
            continue
        view = GameView.from_state(state)
        action = agents[state.current_player()].act(view, rng)
        state.apply_action(action)
    return state.returns()[0]


def head_to_head(
    game, agent_a: Agent, agent_b: Agent, hands: int = 1000, seed: int = 0
) -> MatchResult:
    """Play ``hands`` hands, swapping seats each hand to cancel position.

    Returns agent_a's mean chips/hand (positive = agent_a profits).
    """
    rng = random.Random(seed)
    returns = []
    for h in range(hands):
        state = game.new_initial_state()
        if h % 2 == 0:
            seats = [agent_a, agent_b]       # a is player 0
            r = _play_one(game, state, seats, rng)
        else:
            seats = [agent_b, agent_a]       # a is player 1
            r = -_play_one(game, state, seats, rng)  # flip to a's perspective
        returns.append(r)
    n = len(returns)
    mean = sum(returns) / n
    var = sum((x - mean) ** 2 for x in returns) / n if n > 1 else 0.0
    stderr = (var / n) ** 0.5
    return MatchResult(
        hands=hands,
        agent0=agent_a.name,
        agent1=agent_b.name,
        mean_return=mean,
        stderr=stderr,
    )
