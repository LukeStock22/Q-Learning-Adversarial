"""Entrypoint for running single or comparison experiments."""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .agent import MultiAgentQLearning
from .env import GridworldEnv
from .eval import evaluate, evaluate_with_stats
from .train import moving_average, train
from .utils import ensure_dir, seed_everything

MOVING_AVG_WINDOW = 20
METRICS_FILENAME = "metrics.txt"
CONFIG_USED_FILENAME = "config_used.yaml"


def load_config(path: Path) -> dict:
    """Load YAML config and deep-merge it onto configs/default.yaml when present."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    default_path = Path("configs/default.yaml")
    if path.resolve() == default_path.resolve() or not default_path.exists():
        return config

    with default_path.open("r", encoding="utf-8") as handle:
        default_config = yaml.safe_load(handle) or {}

    return deep_merge(default_config, config)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base and return a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_run_name(config: dict, run_name: str | None) -> dict:
    """Optionally nest outputs under a run-specific subdirectory."""
    if not run_name:
        return config
    project = config.setdefault("project", {})
    base_output = Path(project.get("output_dir", "outputs"))
    project["output_dir"] = str(base_output / run_name)
    return config


def build_env(
    project: dict,
    training: dict,
    scenario_cfg: dict,
    rewards_cfg: dict | None = None,
    state_features_cfg: dict | None = None,
    obstacles: set[tuple[int, int]] | None = None,
    starts: tuple[tuple[int, int], ...] | None = None,
    package_locations: list[tuple[int, int]] | None = None,
    destinations: list[tuple[int, int]] | None = None,
    forklift_starts: list[tuple[int, int]] | None = None,
    adversary_start: tuple[int, int] | None = None,
    adversary_q_table: np.ndarray | None = None,
    evaluation_mode: bool = False,
) -> GridworldEnv:
    """Construct environment from common + scenario-specific config."""
    rewards_cfg = rewards_cfg or {}
    state_features_cfg = state_features_cfg or {}
    return GridworldEnv(
        size=int(project.get("grid_size", 5)),
        max_steps=int(training.get("max_steps", 200)),
        num_packages=int(training.get("num_packages", 1)),
        agent_count=int(training.get("agent_count", 1)),
        obstacles=obstacles,
        starts=starts,
        package_locations=package_locations,
        destinations=destinations,
        forklift_starts=forklift_starts,
        adversary_start=adversary_start,
        scenario=str(scenario_cfg.get("scenario", GridworldEnv.SCENARIO_NATURE)),
        spill_count=int(scenario_cfg.get("spill_count", 0)),
        forklift_count=int(scenario_cfg.get("forklift_count", 1)),
        forklift_move_prob=float(scenario_cfg.get("forklift_move_prob", 0.5)),
        adversary_policy=str(
            scenario_cfg.get("adversary_policy", GridworldEnv.DEFAULT_ADVERSARY_POLICY)
        ),
        adversary_move_prob=float(
            scenario_cfg.get("adversary_move_prob", GridworldEnv.DEFAULT_ADVERSARY_MOVE_PROB)
        ),
        adversary_max_moves=int(
            scenario_cfg.get("adversary_max_moves", GridworldEnv.DEFAULT_ADVERSARY_MAX_MOVES)
        ),
        adversary_learning_alpha=float(
            scenario_cfg.get(
                "adversary_learning_alpha", GridworldEnv.DEFAULT_ADVERSARY_LEARNING_ALPHA
            )
        ),
        adversary_learning_gamma=float(
            scenario_cfg.get(
                "adversary_learning_gamma", GridworldEnv.DEFAULT_ADVERSARY_LEARNING_GAMMA
            )
        ),
        adversary_learning_epsilon_start=float(
            scenario_cfg.get(
                "adversary_learning_epsilon_start",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_EPSILON_START,
            )
        ),
        adversary_learning_epsilon_end=float(
            scenario_cfg.get(
                "adversary_learning_epsilon_end",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_EPSILON_END,
            )
        ),
        adversary_learning_epsilon_decay_episodes=int(
            scenario_cfg.get(
                "adversary_learning_epsilon_decay_episodes",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_EPSILON_DECAY_EPISODES,
            )
        ),
        adversary_learning_progress_reward_scale=float(
            scenario_cfg.get(
                "adversary_learning_progress_reward_scale",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_PROGRESS_REWARD_SCALE,
            )
        ),
        adversary_learning_catch_reward=float(
            scenario_cfg.get(
                "adversary_learning_catch_reward",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_CATCH_REWARD,
            )
        ),
        adversary_learning_objective=str(
            scenario_cfg.get(
                "adversary_learning_objective",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_OBJECTIVE,
            )
        ),
        adversary_freeze_episode=int(
            scenario_cfg.get(
                "adversary_freeze_episode",
                GridworldEnv.DEFAULT_ADVERSARY_FREEZE_EPISODE,
            )
        ),
        adversary_random_tiebreak=bool(scenario_cfg.get("adversary_random_tiebreak", True)),
        adversary_enabled=bool(scenario_cfg.get("adversary_enabled", False)),
        step_penalty=float(rewards_cfg.get("step_penalty", GridworldEnv.DEFAULT_STEP_PENALTY)),
        obstacle_penalty=float(rewards_cfg.get("obstacle_penalty", GridworldEnv.DEFAULT_OBSTACLE_PENALTY)),
        collision_penalty=float(rewards_cfg.get("collision_penalty", GridworldEnv.DEFAULT_COLLISION_PENALTY)),
        forklift_penalty=float(rewards_cfg.get("forklift_penalty", GridworldEnv.DEFAULT_FORKLIFT_PENALTY)),
        adversary_penalty=float(rewards_cfg.get("adversary_penalty", GridworldEnv.DEFAULT_ADVERSARY_PENALTY)),
        pickup_reward=float(rewards_cfg.get("pickup_reward", GridworldEnv.DEFAULT_PICKUP_REWARD)),
        delivery_reward=float(rewards_cfg.get("delivery_reward", GridworldEnv.DEFAULT_DELIVERY_REWARD)),
        distance_shaping_enabled=bool(
            rewards_cfg.get(
                "distance_shaping_enabled", GridworldEnv.DEFAULT_DISTANCE_SHAPING_ENABLED
            )
        ),
        distance_shaping_scale=float(
            rewards_cfg.get(
                "distance_shaping_scale", GridworldEnv.DEFAULT_DISTANCE_SHAPING_SCALE
            )
        ),
        include_relative_package_destination=bool(
            state_features_cfg.get(
                "include_relative_package_destination",
                GridworldEnv.DEFAULT_INCLUDE_RELATIVE_PACKAGE_DESTINATION,
            )
        ),
        adversary_proximity_radius=int(
            state_features_cfg.get(
                "adversary_proximity_radius",
                GridworldEnv.DEFAULT_ADVERSARY_PROXIMITY_RADIUS,
            )
        ),
        include_coarse_destination_direction=bool(
            state_features_cfg.get(
                "include_coarse_destination_direction",
                GridworldEnv.DEFAULT_INCLUDE_COARSE_DESTINATION_DIRECTION,
            )
        ),
        shelf_count=int(training.get("shelf_count", 1)),
        adversary_count=int(scenario_cfg.get("adversary_count", 1)),
        adversary_q_table=adversary_q_table,
        evaluation_mode=evaluation_mode,
    )


def build_agent(env: GridworldEnv, training: dict) -> MultiAgentQLearning:
    """Construct learning agent from config."""
    epsilon_start = float(training.get("epsilon_start", training.get("epsilon", 0.1)))
    return MultiAgentQLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        agent_count=int(training.get("agent_count", 1)),
        alpha=float(training.get("alpha", 0.1)),
        gamma=float(training.get("gamma", 0.95)),
        epsilon=epsilon_start,
        shared_q=bool(training.get("shared_q", True)),
    )


def save_rewards_and_plot(csv_dir: Path, curve_dir: Path, name: str, rewards: list[float]) -> None:
    """Persist reward history and its smoothed curve."""
    avg = moving_average(rewards, window=MOVING_AVG_WINDOW)
    rewards_path = csv_dir / f"{name}_rewards.csv"
    plot_path = curve_dir / f"{name}_learning_curve.png"

    with rewards_path.open("w", encoding="utf-8") as handle:
        handle.write("episode,reward,moving_avg\n")
        for idx, (reward, mean) in enumerate(zip(rewards, avg), start=1):
            handle.write(f"{idx},{reward:.4f},{mean:.4f}\n")

    plt.figure(figsize=(7, 4))
    plt.plot(rewards, label="episode reward", linewidth=1)
    plt.plot(avg, label=f"moving avg ({MOVING_AVG_WINDOW})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Training rewards ({name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def save_used_config(output_root: Path, config: dict) -> None:
    """Persist the merged config used for a run alongside its outputs."""
    config_path = output_root / CONFIG_USED_FILENAME
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def save_rollout_gif(
    env: GridworldEnv,
    agent: MultiAgentQLearning,
    gif_path: Path,
    max_steps: int,
    frame_duration: float = 0.6,
) -> None:
    """Generate a greedy rollout GIF for one environment/policy pair."""
    frames = []
    state = env.reset()
    done = False
    step = 0

    with tempfile.TemporaryDirectory(prefix="qlearn_frames_") as tmp_dir:
        while not done and step < max_steps:
            frame_path = Path(tmp_dir) / f"step_{step:03d}.png"
            env.render(path=str(frame_path))
            frames.append(imageio.imread(frame_path))

            state_idx = env.encode_state(state)
            actions = agent.select_greedy_actions(state_idx)
            joint_action = 0
            for idx, action in enumerate(actions):
                joint_action += action * (env.n_actions**idx)

            result = env.step(joint_action)
            state = result.state
            done = result.done
            step += 1

        # Render terminal state so successful delivery appears in the GIF.
        final_frame_path = Path(tmp_dir) / f"step_{step:03d}_final.png"
        env.render(path=str(final_frame_path))
        frames.append(imageio.imread(final_frame_path))

    if frames:
        imageio.mimsave(gif_path, frames, duration=frame_duration)


def run_comparison(config: dict) -> None:
    """Train Nature and Adversary policies and evaluate both cross-scenario."""
    project = config["project"]
    run = config.get("run", {})
    training = config["training"]
    eval_cfg = config.get("eval", {})
    rewards_cfg = config.get("rewards", {})
    state_features_cfg = config.get("state_features", {})
    scenarios = config["scenarios"]
    rewards_cfg = config.get("rewards", {})

    tier = run.get("tier", "debug")
    tier_cfg = run.get(tier, {})
    output_root = ensure_dir(project.get("output_dir", "outputs"))
    csv_dir = ensure_dir(output_root / "csv")
    pkl_dir = ensure_dir(output_root / "pkl")
    curve_dir = ensure_dir(output_root / "learningcurves")
    png_dir = ensure_dir(output_root / "png")
    gif_dir = ensure_dir(output_root / "gif")
    txt_dir = ensure_dir(output_root / "txt")
    save_used_config(output_root, config)

    train_episodes = int(tier_cfg.get("train_episodes", 5000))
    eval_episodes = int(tier_cfg.get("eval_episodes", 50))
    eval_max_steps = int(eval_cfg.get("max_steps", training.get("max_steps", 200)))
    ood_layout_count = max(1, int(eval_cfg.get("ood_layout_count", 1)))

    base_seed = int(project.get("seed", 7))

    def sample_layout_bundle(seed: int) -> dict:
        """Sample one fully specified layout bundle shared across scenarios."""
        seed_everything(seed)
        nature_cfg = scenarios.get(GridworldEnv.SCENARIO_NATURE, {})
        shelf_sampler = build_env(
            project,
            training,
            nature_cfg,
            rewards_cfg=rewards_cfg,
            state_features_cfg=state_features_cfg,
        )
        shelves = set(shelf_sampler.shelf_obstacles)
        layout_sampler = build_env(
            project,
            training,
            nature_cfg,
            rewards_cfg=rewards_cfg,
            state_features_cfg=state_features_cfg,
            obstacles=set(shelves),
        )
        layout_sampler.reset()
        forklift_starts = list(layout_sampler.forklift_positions)
        adversary_start = forklift_starts[0] if forklift_starts else None
        return {
            "obstacles": shelves,
            "starts": tuple(layout_sampler.starts),
            "package_locations": list(layout_sampler.package_locations),
            "destinations": list(layout_sampler.destinations),
            "forklift_starts": forklift_starts,
            "adversary_start": adversary_start,
        }

    num_train_layouts = max(1, int(training.get("num_train_layouts", 1)))

    layout_train = sample_layout_bundle(base_seed)
    extra_train_layouts = [
        sample_layout_bundle(base_seed + 10000 + i) for i in range(num_train_layouts - 1)
    ]
    all_train_layouts = [layout_train] + extra_train_layouts

    layout_ood_bundles = [
        sample_layout_bundle(base_seed + 5000 + idx) for idx in range(ood_layout_count)
    ]

    policy_agents: dict[str, MultiAgentQLearning] = {}
    train_rewards_by_policy: dict[str, list[float]] = {}
    learned_adversary_eval_tables: dict[str, dict[str, np.ndarray]] = {}
    scenario_order = [GridworldEnv.SCENARIO_NATURE, GridworldEnv.SCENARIO_ADVERSARY]

    for idx, scenario_name in enumerate(scenario_order):
        seed_everything(base_seed + idx * 100)
        train_envs = [
            build_env(
                project,
                training,
                scenarios.get(scenario_name, {}),
                rewards_cfg=rewards_cfg,
                state_features_cfg=state_features_cfg,
                obstacles=set(layout_bundle["obstacles"]),
                starts=layout_bundle["starts"],
                package_locations=layout_bundle["package_locations"],
                destinations=layout_bundle["destinations"],
                forklift_starts=layout_bundle["forklift_starts"],
                adversary_start=layout_bundle["adversary_start"],
            )
            for layout_bundle in all_train_layouts
        ]
        env = train_envs[0]
        agent = build_agent(env, training)

        print(f"Training policy on '{scenario_name}' scenario ({num_train_layouts} layout(s))")
        rewards = train(
            train_envs if num_train_layouts > 1 else env,
            agent,
            episodes=train_episodes,
            max_steps=int(training.get("max_steps", 200)),
            progress_every=int(training.get("progress_every", 100)),
            output_dir=str(pkl_dir),
            artifact_prefix=scenario_name,
            epsilon_start=float(training.get("epsilon_start", training.get("epsilon", 0.1))),
            epsilon_end=float(training.get("epsilon_end", training.get("epsilon", 0.1))),
        )
        save_rewards_and_plot(csv_dir, curve_dir, scenario_name, rewards)
        train_rewards_by_policy[scenario_name] = rewards

        env.reset()
        env.render(path=str(png_dir / f"{scenario_name}_layout.png"))

        policy_agents[scenario_name] = agent
        if (
            scenario_name == GridworldEnv.SCENARIO_ADVERSARY
            and env.adversary_policy == GridworldEnv.ADVERSARY_POLICY_LEARNING
        ):
            adversary_tables = [train_env.get_adversary_q_table() for train_env in train_envs]
            learned_adversary_eval_tables[scenario_name] = {
                "id_a": np.array(adversary_tables[0], copy=True),
                # OOD layouts have no one-to-one trained adversary table, so use the
                # average learned policy across the multi-layout curriculum.
                "ood": np.mean(np.stack(adversary_tables, axis=0), axis=0),
            }
        print(f"{scenario_name} final reward: {rewards[-1]:.2f}")

    def build_eval_env_for_layout(
        layout_bundle: dict,
        eval_name: str,
        split: str,
    ) -> GridworldEnv:
        """Construct an eval env, reusing learned adversary state when applicable."""
        eval_scenario_cfg = scenarios.get(eval_name, {})
        adversary_q_table = None
        evaluation_mode = False
        if (
            eval_name == GridworldEnv.SCENARIO_ADVERSARY
            and str(
                eval_scenario_cfg.get(
                    "adversary_policy",
                    GridworldEnv.DEFAULT_ADVERSARY_POLICY,
                )
            )
            == GridworldEnv.ADVERSARY_POLICY_LEARNING
        ):
            tables = learned_adversary_eval_tables.get(eval_name)
            if tables is not None:
                adversary_q_table = tables[split]
                evaluation_mode = True

        return build_env(
            project,
            training,
            eval_scenario_cfg,
            rewards_cfg=rewards_cfg,
            state_features_cfg=state_features_cfg,
            obstacles=set(layout_bundle["obstacles"]),
            starts=layout_bundle["starts"],
            package_locations=layout_bundle["package_locations"],
            destinations=layout_bundle["destinations"],
            forklift_starts=layout_bundle["forklift_starts"],
            adversary_start=layout_bundle["adversary_start"],
            adversary_q_table=adversary_q_table,
            evaluation_mode=evaluation_mode,
        )

    def evaluate_matrix(
        layout_bundles: list[dict],
        seed_base: int,
        split: str,
    ) -> list[tuple[str, str, float, int, int, float]]:
        """Cross-test all policies across one or more layout bundles."""
        matrix: list[tuple[str, str, float, int, int, float]] = []
        for train_idx, train_name in enumerate(scenario_order):
            for eval_idx, eval_name in enumerate(scenario_order):
                total_reward = 0.0
                total_episodes = 0
                total_collisions = 0
                total_delivered = 0
                total_steps = 0
                for layout_idx, layout_bundle in enumerate(layout_bundles):
                    # Include train_idx so stochastic tie-breaks do not replay the exact
                    # same RNG stream across different trained policies.
                    seed_everything(seed_base + train_idx * 100000 + eval_idx * 1000 + layout_idx)
                    eval_env = build_eval_env_for_layout(layout_bundle, eval_name, split)
                    stats = evaluate_with_stats(
                        eval_env,
                        policy_agents[train_name],
                        episodes=eval_episodes,
                        max_steps=eval_max_steps,
                    )
                    total_reward += sum(stats["rewards"])
                    total_episodes += int(stats["episodes"])
                    total_collisions += int(stats["collisions"])
                    total_delivered += int(stats["delivered_packages"])
                    total_steps += int(stats["total_steps"])
                avg_reward = total_reward / max(1, total_episodes)
                avg_steps = total_steps / max(1, total_episodes)
                matrix.append(
                    (
                        train_name,
                        eval_name,
                        avg_reward,
                        total_collisions,
                        total_delivered,
                        avg_steps,
                    )
                )
        return matrix

    results_id_a = evaluate_matrix([layout_train], base_seed + 1000, "id_a")
    results_ood = evaluate_matrix(layout_ood_bundles, base_seed + 2000, "ood")

    matrix_ida_path = csv_dir / "strategy_comparison_ida.csv"
    with matrix_ida_path.open("w", encoding="utf-8") as handle:
        handle.write("train_policy,eval_scenario,avg_reward,collisions,delivered_packages,avg_steps\n")
        for train_name, eval_name, avg_reward, collisions, delivered, avg_steps in results_id_a:
            handle.write(f"{train_name},{eval_name},{avg_reward:.4f},{collisions},{delivered},{avg_steps:.4f}\n")

    matrix_ood_path = csv_dir / "strategy_comparison_ood_layout.csv"
    with matrix_ood_path.open("w", encoding="utf-8") as handle:
        handle.write("train_policy,eval_scenario,avg_reward,collisions,delivered_packages,avg_steps\n")
        for train_name, eval_name, avg_reward, collisions, delivered, avg_steps in results_ood:
            handle.write(f"{train_name},{eval_name},{avg_reward:.4f},{collisions},{delivered},{avg_steps:.4f}\n")

    metrics_path = txt_dir / METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write("experiment_mode=comparison\n")
        handle.write(f"tier={tier}\n")
        handle.write(f"train_episodes={train_episodes}\n")
        handle.write(f"eval_episodes={eval_episodes}\n")
        handle.write(f"ood_layout_count={ood_layout_count}\n")
        for train_name, eval_name, avg_reward, collisions, delivered, avg_steps in results_id_a:
            handle.write(f"id_a_{train_name}_on_{eval_name}_reward={avg_reward:.4f}\n")
            handle.write(f"id_a_{train_name}_on_{eval_name}_collisions={collisions}\n")
            handle.write(f"id_a_{train_name}_on_{eval_name}_delivered={delivered}\n")
            handle.write(f"id_a_{train_name}_on_{eval_name}_avg_steps={avg_steps:.4f}\n")
        for train_name, eval_name, avg_reward, collisions, delivered, avg_steps in results_ood:
            handle.write(f"ood_layout_{train_name}_on_{eval_name}_reward={avg_reward:.4f}\n")
            handle.write(f"ood_layout_{train_name}_on_{eval_name}_collisions={collisions}\n")
            handle.write(f"ood_layout_{train_name}_on_{eval_name}_delivered={delivered}\n")
            handle.write(f"ood_layout_{train_name}_on_{eval_name}_avg_steps={avg_steps:.4f}\n")

    # Restore old-style summary metrics, per training strategy.
    result_lookup_ida = {(train_name, eval_name): avg_reward for train_name, eval_name, avg_reward, _, _, _ in results_id_a}
    result_lookup_ood = {(train_name, eval_name): avg_reward for train_name, eval_name, avg_reward, _, _, _ in results_ood}
    print("\n=== Summary (Old-style Metrics) ===")
    for scenario_name in scenario_order:
        rewards = train_rewards_by_policy[scenario_name]
        moving = moving_average(rewards, window=MOVING_AVG_WINDOW)
        id_a = result_lookup_ida[(scenario_name, scenario_name)]
        ood = result_lookup_ood[(scenario_name, scenario_name)]

        print(f"\n[{scenario_name.upper()} POLICY]")
        print(f"Final episode reward: {rewards[-1]:.2f}")
        print(f"Last-20 moving average: {moving[-1]:.2f}")
        print(f"ID-A eval average reward: {id_a:.2f}")
        print(f"OOD eval average reward: {ood:.2f}")

    id_a_episode_count = eval_episodes
    ood_episode_count = eval_episodes * ood_layout_count

    print("\n=== ID-A Cross-Scenario Matrix ===")
    for train_name, eval_name, avg_reward, collisions, delivered, avg_steps in results_id_a:
        print(
            f"Policy trained on {train_name} evaluated on {eval_name}: "
            f"{avg_reward:.2f} | collisions: {collisions}/{id_a_episode_count}"
            f" ({collisions / max(1, id_a_episode_count):.2f}/ep)"
            f" | packages delivered: {delivered}/{id_a_episode_count}"
            f" ({delivered / max(1, id_a_episode_count):.2f}/ep)"
            f" | avg steps: {avg_steps:.2f}"
        )

    print(f"\n=== OOD-Layout Cross-Scenario Matrix ({ood_layout_count} layouts) ===")
    for train_name, eval_name, avg_reward, collisions, delivered, avg_steps in results_ood:
        print(
            f"Policy trained on {train_name} evaluated on {eval_name}: "
            f"{avg_reward:.2f} | collisions: {collisions}/{ood_episode_count}"
            f" ({collisions / max(1, ood_episode_count):.2f}/ep)"
            f" | packages delivered: {delivered}/{ood_episode_count}"
            f" ({delivered / max(1, ood_episode_count):.2f}/ep)"
            f" | avg steps: {avg_steps:.2f}"
        )

    # Write representative rollout GIFs for ID-A and OOD-layout cross tests.
    layout_ood = layout_ood_bundles[0]
    for train_name in scenario_order:
        for eval_name in scenario_order:
            ida_env = build_eval_env_for_layout(layout_train, eval_name, "id_a")
            save_rollout_gif(
                ida_env,
                policy_agents[train_name],
                gif_dir / f"ida_{train_name}_on_{eval_name}.gif",
                max_steps=eval_max_steps,
            )

            ood_env = build_eval_env_for_layout(layout_ood, eval_name, "ood")
            save_rollout_gif(
                ood_env,
                policy_agents[train_name],
                gif_dir / f"ood_{train_name}_on_{eval_name}.gif",
                max_steps=eval_max_steps,
            )

    print(f"Wrote comparison outputs to {output_root}")


def run_single(config: dict) -> None:
    """Run one training scenario for quick local iteration."""
    project = config["project"]
    run = config.get("run", {})
    training = config["training"]
    eval_cfg = config.get("eval", {})
    rewards_cfg = config.get("rewards", {})
    state_features_cfg = config.get("state_features", {})
    output_root = ensure_dir(project.get("output_dir", "outputs"))
    csv_dir = ensure_dir(output_root / "csv")
    pkl_dir = ensure_dir(output_root / "pkl")
    curve_dir = ensure_dir(output_root / "learningcurves")
    png_dir = ensure_dir(output_root / "png")
    gif_dir = ensure_dir(output_root / "gif")
    save_used_config(output_root, config)

    tier = run.get("tier", "debug")
    tier_cfg = run.get(tier, {})

    scenario_cfg = {
        "scenario": str(training.get("scenario", GridworldEnv.SCENARIO_NATURE)),
        "spill_count": int(training.get("spill_count", 0)),
        "forklift_count": int(training.get("forklift_count", 1)),
        "forklift_move_prob": float(training.get("forklift_move_prob", 0.5)),
        "adversary_policy": str(
            training.get("adversary_policy", GridworldEnv.DEFAULT_ADVERSARY_POLICY)
        ),
        "adversary_move_prob": float(
            training.get("adversary_move_prob", GridworldEnv.DEFAULT_ADVERSARY_MOVE_PROB)
        ),
        "adversary_max_moves": int(
            training.get("adversary_max_moves", GridworldEnv.DEFAULT_ADVERSARY_MAX_MOVES)
        ),
        "adversary_learning_alpha": float(
            training.get(
                "adversary_learning_alpha", GridworldEnv.DEFAULT_ADVERSARY_LEARNING_ALPHA
            )
        ),
        "adversary_learning_gamma": float(
            training.get(
                "adversary_learning_gamma", GridworldEnv.DEFAULT_ADVERSARY_LEARNING_GAMMA
            )
        ),
        "adversary_learning_epsilon_start": float(
            training.get(
                "adversary_learning_epsilon_start",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_EPSILON_START,
            )
        ),
        "adversary_learning_epsilon_end": float(
            training.get(
                "adversary_learning_epsilon_end",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_EPSILON_END,
            )
        ),
        "adversary_learning_epsilon_decay_episodes": int(
            training.get(
                "adversary_learning_epsilon_decay_episodes",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_EPSILON_DECAY_EPISODES,
            )
        ),
        "adversary_learning_progress_reward_scale": float(
            training.get(
                "adversary_learning_progress_reward_scale",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_PROGRESS_REWARD_SCALE,
            )
        ),
        "adversary_learning_catch_reward": float(
            training.get(
                "adversary_learning_catch_reward",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_CATCH_REWARD,
            )
        ),
        "adversary_learning_objective": str(
            training.get(
                "adversary_learning_objective",
                GridworldEnv.DEFAULT_ADVERSARY_LEARNING_OBJECTIVE,
            )
        ),
        "adversary_freeze_episode": int(
            training.get(
                "adversary_freeze_episode",
                GridworldEnv.DEFAULT_ADVERSARY_FREEZE_EPISODE,
            )
        ),
        "adversary_random_tiebreak": bool(training.get("adversary_random_tiebreak", True)),
        "adversary_enabled": bool(training.get("adversary_enabled", False)),
    }

    seed_everything(int(project.get("seed", 7)))
    env = build_env(
        project,
        training,
        scenario_cfg,
        rewards_cfg=rewards_cfg,
        state_features_cfg=state_features_cfg,
    )
    agent = build_agent(env, training)

    rewards = train(
        env,
        agent,
        episodes=int(tier_cfg.get("train_episodes", 5000)),
        max_steps=int(training.get("max_steps", 200)),
        progress_every=int(training.get("progress_every", 100)),
        output_dir=str(pkl_dir),
        epsilon_start=float(training.get("epsilon_start", training.get("epsilon", 0.1))),
        epsilon_end=float(training.get("epsilon_end", training.get("epsilon", 0.1))),
    )
    save_rewards_and_plot(csv_dir, curve_dir, "single", rewards)

    eval_rewards = evaluate(
        env,
        agent,
        episodes=int(tier_cfg.get("eval_episodes", 50)),
        max_steps=int(eval_cfg.get("max_steps", 200)),
    )

    env.reset()
    env.render(path=str(png_dir / "single_layout.png"))

    print(f"Final episode reward: {rewards[-1]:.2f}")
    print(f"Eval average reward: {sum(eval_rewards) / len(eval_rewards):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q-learning experiments")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional subfolder name under project.output_dir for this run",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    config = apply_run_name(config, args.run_name or None)
    mode = config.get("experiment", {}).get("mode", "comparison")

    if mode == "single":
        run_single(config)
        return

    run_comparison(config)


if __name__ == "__main__":
    main()
