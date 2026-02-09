"""Training loop.

Runs multiple episodes of interaction between the agent and environment,
and updates the Q-table after every step.
"""

from __future__ import annotations

from typing import Iterable

from .agent import QLearningAgent
from .env import GridworldEnv


def train(
    env: GridworldEnv,
    agent: QLearningAgent,
    episodes: int = 200,
    max_steps: int = 200,
) -> list[float]:
    """Train the agent and return total reward per episode."""
    rewards: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        done = False

        for _ in range(max_steps):
            # Convert (row, col) to a single table index.
            state_idx = env.encode_state(state)
            # Choose an action (explore or exploit).
            action = agent.select_action(state_idx)
            # Apply the action in the environment.
            result = env.step(action)
            # Encode next state for indexing.
            next_state_idx = env.encode_state(result.state)

            # Learn from the transition.
            agent.update(state_idx, action, result.reward, next_state_idx, result.done)

            total += result.reward
            state = result.state
            done = result.done
            if done:
                break

        rewards.append(total)

    return rewards


def moving_average(values: Iterable[float], window: int = 10) -> list[float]:
    """Simple moving average for smoothing reward curves."""
    values = list(values)
    if window <= 1:
        return values
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out
