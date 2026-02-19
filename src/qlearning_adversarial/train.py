"""Training loop.

Runs multiple episodes of interaction between the agent and environment,
and updates the Q-table after every step.
"""

from __future__ import annotations

import pickle 
import os    

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
) -> list[float]:
    """Train the agent and return total reward per episode."""
    rewards: list[float] = []

    for episode_idx in range(1, episodes + 1):
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


    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    

    q_table_path = os.path.join(output_dir, "q_table.pkl")

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
        "agent_count": env.agent_count,
        "spill_count": env.spill_count,
        "starts": env.starts,
        "package_locations": env.package_locations,
        "destinations": env.destinations,
        "obstacles": env.shelf_obstacles, 
        "max_steps": env.max_steps
    }
    config_path = os.path.join(output_dir, "env_config.pkl")
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