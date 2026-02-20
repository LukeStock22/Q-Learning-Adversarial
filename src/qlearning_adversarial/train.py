"""Training loop.

Runs multiple episodes of interaction between the agent and environment,
and updates the Q-table after every step.
"""

from __future__ import annotations

import os
import pickle

from typing import Iterable

from .agent import MultiAgentQLearning
from .env import GridworldEnv


DEFAULT_PROGRESS_EVERY = 100
DEFAULT_AGENT_COUNT = 1


def train(
    env: GridworldEnv,
    agent: MultiAgentQLearning,
    episodes: int = 200,
    max_steps: int = 200,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    output_dir: str = "outputs",
    artifact_prefix: str = "",
    epsilon_start: float | None = None,
    epsilon_end: float | None = None,
) -> list[float]:
    """Train the agent and return total reward per episode."""
    rewards: list[float] = []

    for episode_idx in range(1, episodes + 1):
        # Optional linear epsilon decay across training episodes.
        if epsilon_start is not None and epsilon_end is not None:
            if episodes <= 1:
                agent.epsilon = float(epsilon_end)
            else:
                progress = (episode_idx - 1) / (episodes - 1)
                agent.epsilon = float(epsilon_start + progress * (epsilon_end - epsilon_start))

        state = env.reset()
        total = 0.0
        done = False

        for _ in range(max_steps):
            # Convert (row, col) to a single table index.
            state_idx = env.encode_state(state)
            # Choose actions for both agents (explore or exploit).
            actions = agent.select_actions(state_idx)
            joint_action = 0
            for idx, action in enumerate(actions):
                joint_action += action * (env.n_actions**idx)
            # Apply the action in the environment.
            result = env.step(joint_action)
            # Encode next state for indexing.
            next_state_idx = env.encode_state(result.state)

            # Learn from the transition.
            agent.update(state_idx, actions, result.reward, next_state_idx, result.done)

            total += result.reward
            state = result.state
            done = result.done
            if done:
                break

        rewards.append(total)

        if progress_every > 0 and episode_idx % progress_every == 0:
            print(".", end="", flush=True)

    if progress_every > 0:
        print()


    os.makedirs(output_dir, exist_ok=True)
    

    prefix = f"{artifact_prefix}_" if artifact_prefix else ""
    q_table_path = os.path.join(output_dir, f"{prefix}q_table.pkl")

    if hasattr(agent, "q_shared") and agent.q_shared is not None:
        with open(q_table_path, "wb") as f:
            pickle.dump(agent.q_shared, f)
        print(f"\n[System] Shared Q-Table saved to: {q_table_path}")
    elif hasattr(agent, "q_agents") and agent.q_agents is not None:
        with open(q_table_path, "wb") as f:
            pickle.dump(agent.q_agents, f)
        print(f"\n[System] Multi-Agent Q-Tables saved to: {q_table_path}")
    elif hasattr(agent, "q"):
        with open(q_table_path, "wb") as f:
            pickle.dump(agent.q, f)
        print(f"\n[System] Single Agent Q-Table saved to: {q_table_path}")
    else:
        print(f"\n[Warning] Could not save model (no known Q-table attribute).")


    env_config = {
        "size": env.size,
        "scenario": env.scenario,
        "agent_count": env.agent_count,
        "forklift_count": env.forklift_count,
        "forklift_move_prob": env.forklift_move_prob,
        "adversary_policy": env.adversary_policy,
        "adversary_move_prob": env.adversary_move_prob,
        "adversary_max_moves": env.adversary_max_moves,
        "adversary_learning_alpha": env.adversary_learning_alpha,
        "adversary_learning_gamma": env.adversary_learning_gamma,
        "adversary_learning_epsilon_start": env.adversary_learning_epsilon_start,
        "adversary_learning_epsilon_end": env.adversary_learning_epsilon_end,
        "adversary_learning_epsilon_decay_episodes": env.adversary_learning_epsilon_decay_episodes,
        "adversary_learning_progress_reward_scale": env.adversary_learning_progress_reward_scale,
        "adversary_learning_catch_reward": env.adversary_learning_catch_reward,
        "adversary_learning_objective": env.adversary_learning_objective,
        "adversary_enabled": env.adversary_enabled,
        "adversary_random_tiebreak": env.adversary_random_tiebreak,
        "step_penalty": env.step_penalty,
        "obstacle_penalty": env.obstacle_penalty,
        "collision_penalty": env.collision_penalty,
        "forklift_penalty": env.forklift_penalty,
        "adversary_penalty": env.adversary_penalty,
        "pickup_reward": env.pickup_reward,
        "delivery_reward": env.delivery_reward,
        "distance_shaping_enabled": env.distance_shaping_enabled,
        "distance_shaping_scale": env.distance_shaping_scale,
        "include_relative_package_destination": env.include_relative_package_destination,
        "spill_count": env.spill_count,
        "starts": env.starts,
        "package_locations": env.package_locations,
        "destinations": env.destinations,
        "forklift_starts": env.fixed_forklift_starts,
        "adversary_start": env.fixed_adversary_start,
        "obstacles": env.shelf_obstacles,
        "max_steps": env.max_steps,
    }
    config_path = os.path.join(output_dir, f"{prefix}env_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(env_config, f)
    print(f"[System] Environment Layout saved to: {config_path}")
    # ===================================================================

    return rewards


DEFAULT_MOVING_AVG_WINDOW = 10


def moving_average(values: Iterable[float], window: int = DEFAULT_MOVING_AVG_WINDOW) -> list[float]:
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
