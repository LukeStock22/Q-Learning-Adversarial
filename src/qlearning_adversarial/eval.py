"""Evaluation utilities."""

from __future__ import annotations

from .agent import QLearningAgent
from .env import GridworldEnv


def evaluate(
    env: GridworldEnv,
    agent: QLearningAgent,
    episodes: int = 50,
    max_steps: int = 200,
) -> list[float]:
    """Run evaluation episodes using greedy actions and return total rewards."""
    rewards: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        total = 0.0

        for _ in range(max_steps):
            state_idx = env.encode_state(state)
            action = agent.select_greedy_action(state_idx)
            result = env.step(action)
            total += result.reward
            state = result.state
            if result.done:
                break

        rewards.append(total)

    return rewards
