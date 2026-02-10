"""Entrypoint for running training or evaluation.

This file wires together the environment, the agent, the training loop,
and logging to disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from .agent import MultiAgentQLearning
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

    MOVING_AVG_WINDOW = 20
    METRICS_FILENAME = "metrics.txt"
    REWARDS_FILENAME = "rewards.csv"
    PLOT_FILENAME = "learning_curve.png"
    GRID_FILENAME = "gridworld_layout.png"

    project = config["project"]
    run = config.get("run", {})
    training = config["training"]
    evaluation = config["eval"]

    tier = run.get("tier", "debug")
    tier_cfg = run.get(tier, {})

    # Fix random seeds for reproducible results.
    base_seed = int(project.get("seed", 7))
    seed_everything(base_seed)

    # Build environment and agent from config.
    env = GridworldEnv(
        size=int(project.get("grid_size", 10)),
        max_steps=int(training.get("max_steps", 200)),
        num_packages=int(training.get("num_packages", 3)),
        spill_count=int(training.get("spill_count", 2)),
        agent_count=int(training.get("agent_count", 1)),
    )
    agent = MultiAgentQLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        agent_count=int(training.get("agent_count", 1)),
        alpha=float(training.get("alpha", 0.1)),
        gamma=float(training.get("gamma", 0.95)),
        epsilon=float(training.get("epsilon", 0.1)),
        shared_q=bool(training.get("shared_q", True)),
    )

    # Train agent.
    rewards = train(
        env,
        agent,
        episodes=int(tier_cfg.get("train_episodes", 5000)),
        max_steps=int(training.get("max_steps", 200)),
        progress_every=int(training.get("progress_every", 100)),
    )
    avg = moving_average(rewards, window=MOVING_AVG_WINDOW)

    eval_mode = evaluation.get("mode", "both")
    eval_id_rewards: list[float] = []
    eval_ood_rewards: list[float] = []

    # ID-A evaluation: same exact layout, new per-episode spills.
    if eval_mode in ("both", "id_a", "id"):
        eval_id_rewards = evaluate(
            env,
            agent,
            episodes=int(tier_cfg.get("eval_id_episodes", 50)),
            max_steps=int(evaluation.get("max_steps", 200)),
        )

    # OOD evaluation: new shelf + new package/destination layout.
    if eval_mode in ("both", "ood"):
        seed_everything(base_seed + 1)
        ood_env = GridworldEnv(
            size=int(project.get("grid_size", 10)),
            max_steps=int(training.get("max_steps", 200)),
            num_packages=int(training.get("num_packages", 3)),
            spill_count=int(training.get("spill_count", 2)),
            agent_count=int(training.get("agent_count", 1)),
        )
        eval_ood_rewards = evaluate(
            ood_env,
            agent,
            episodes=int(tier_cfg.get("eval_ood_episodes", 50)),
            max_steps=int(evaluation.get("max_steps", 200)),
        )

    # Log outputs.
    output_dir = ensure_dir(project.get("output_dir", "outputs"))
    rewards_path = output_dir / REWARDS_FILENAME
    plot_path = output_dir / PLOT_FILENAME
    grid_path = output_dir / GRID_FILENAME
    metrics_path = output_dir / METRICS_FILENAME

    with rewards_path.open("w", encoding="utf-8") as handle:
        handle.write("episode,reward,moving_avg\n")
        for idx, (reward, mean) in enumerate(zip(rewards, avg), start=1):
            handle.write(f"{idx},{reward:.4f},{mean:.4f}\n")

    plt.figure(figsize=(7, 4))
    plt.plot(rewards, label="episode reward", linewidth=1)
    plt.plot(avg, label=f"moving avg ({MOVING_AVG_WINDOW})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Save a static image of the current gridworld layout.
    env.reset()
    env.render(path=str(grid_path))

    # Print a minimal summary of progress.
    print(f"Final episode reward: {rewards[-1]:.2f}")
    print(f"Last-20 moving average: {avg[-1]:.2f}")
    if eval_id_rewards:
        print(f"ID-A eval average reward: {sum(eval_id_rewards) / len(eval_id_rewards):.2f}")
    if eval_ood_rewards:
        print(f"OOD eval average reward: {sum(eval_ood_rewards) / len(eval_ood_rewards):.2f}")
    print(f"Wrote {rewards_path}, {plot_path}, {grid_path}, and {metrics_path}")

    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(f"tier={tier}\n")
        if eval_id_rewards:
            handle.write(f"id_a_avg_reward={sum(eval_id_rewards) / len(eval_id_rewards):.4f}\n")
        if eval_ood_rewards:
            handle.write(f"ood_avg_reward={sum(eval_ood_rewards) / len(eval_ood_rewards):.4f}\n")


if __name__ == "__main__":
    main()
