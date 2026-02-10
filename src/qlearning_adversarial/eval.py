"""Evaluation utilities."""

from __future__ import annotations

from .agent import MultiAgentQLearning
from .env import GridworldEnv


def evaluate(
    env: GridworldEnv,
    agent: MultiAgentQLearning,
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
            actions = agent.select_greedy_actions(state_idx)
            joint_action = 0
            for idx, action in enumerate(actions):
                joint_action += action * (env.n_actions**idx)
            result = env.step(joint_action)
            total += result.reward
            state = result.state
            if result.done:
                break

        rewards.append(total)

    return rewards
