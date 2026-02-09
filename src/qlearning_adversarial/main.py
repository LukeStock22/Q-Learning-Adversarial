"""Entrypoint for running training or evaluation.

This file wires together the environment, the agent, the training loop,
and logging to disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from .agent import QLearningAgent
from .env import GridworldEnv
from .eval import evaluate
from .train import moving_average, train
from .utils import ensure_dir, seed_everything


def load_config(path: Path) -> dict:
    """Load YAML configuration into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    config = load_config(Path("configs/default.yaml"))

    project = config["project"]
    training = config["training"]
    evaluation = config["eval"]

    # Fix random seeds for reproducible results.
    seed_everything(int(project.get("seed", 7)))

    # Build environment and agent from config.
    env = GridworldEnv(
        size=int(project.get("grid_size", 10)),
        max_steps=int(training.get("max_steps", 200)),
    )
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=float(training.get("alpha", 0.1)),
        gamma=float(training.get("gamma", 0.95)),
        epsilon=float(training.get("epsilon", 0.1)),
    )

    # Train agent.
    rewards = train(
        env,
        agent,
        episodes=int(training.get("episodes", 1000)),
        max_steps=int(training.get("max_steps", 200)),
    )
    avg = moving_average(rewards, window=20)

    # Evaluate agent with greedy actions.
    eval_rewards = evaluate(
        env,
        agent,
        episodes=int(evaluation.get("episodes", 50)),
        max_steps=int(evaluation.get("max_steps", 200)),
    )

    # Log outputs.
    output_dir = ensure_dir(project.get("output_dir", "outputs"))
    rewards_path = output_dir / "rewards.csv"
    plot_path = output_dir / "learning_curve.png"

    with rewards_path.open("w", encoding="utf-8") as handle:
        handle.write("episode,reward,moving_avg\n")
        for idx, (reward, mean) in enumerate(zip(rewards, avg), start=1):
            handle.write(f"{idx},{reward:.4f},{mean:.4f}\n")

    plt.figure(figsize=(7, 4))
    plt.plot(rewards, label="episode reward", linewidth=1)
    plt.plot(avg, label="moving avg (20)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Print a minimal summary of progress.
    print(f"Final episode reward: {rewards[-1]:.2f}")
    print(f"Last-20 moving average: {avg[-1]:.2f}")
    print(f"Eval average reward: {sum(eval_rewards) / len(eval_rewards):.2f}")
    print(f"Wrote {rewards_path} and {plot_path}")


if __name__ == "__main__":
    main()
