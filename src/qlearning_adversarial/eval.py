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


def evaluate_with_stats(
    env: GridworldEnv,
    agent: MultiAgentQLearning,
    episodes: int = 50,
    max_steps: int = 200,
) -> dict:
    """Run evaluation and return reward list plus aggregate outcome statistics."""
    rewards: list[float] = []
    collisions = 0
    delivered_packages = 0
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        done = False
        last_info: dict = {}
        episode_steps = 0

        for _ in range(max_steps):
            state_idx = env.encode_state(state)
            actions = agent.select_greedy_actions(state_idx)
            joint_action = 0
            for idx, action in enumerate(actions):
                joint_action += action * (env.n_actions**idx)
            result = env.step(joint_action)
            total += result.reward
            state = result.state
            last_info = result.info
            episode_steps += 1
            if result.done:
                done = True
                break

        rewards.append(total)
        total_steps += episode_steps
        if done and "caught" in last_info:
            collisions += 1
        delivered_packages += env.delivered_package_count()

    return {
        "rewards": rewards,
        "collisions": collisions,
        "delivered_packages": delivered_packages,
        "total_steps": total_steps,
        "avg_steps": total_steps / max(1, episodes),
        "episodes": episodes,
    }
