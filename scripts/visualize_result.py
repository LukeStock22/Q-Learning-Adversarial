import argparse
import os
import pickle
import shutil

import imageio.v2 as imageio
import yaml

from qlearning_adversarial.agent import MultiAgentQLearning
from qlearning_adversarial.env import GridworldEnv


def run_visualization(prefix: str = ""):
    output_dir = "outputs"
    if os.path.exists("configs/default.yaml"):
        with open("configs/default.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
            output_dir = config.get("project", {}).get("output_dir", "outputs")

    pkl_dir = os.path.join(output_dir, "pkl")
    gif_dir = os.path.join(output_dir, "gif")
    os.makedirs(gif_dir, exist_ok=True)

    prefix_part = f"{prefix}_" if prefix else ""
    config_path = os.path.join(pkl_dir, f"{prefix_part}env_config.pkl")
    q_table_path = os.path.join(pkl_dir, f"{prefix_part}q_table.pkl")

    if os.path.exists(config_path):
        print(f"Loading saved map configuration: {config_path}")
        with open(config_path, "rb") as f:
            env_config = pickle.load(f)
        env = GridworldEnv(**env_config)
    else:
        print("Map configuration file not found. Cannot visualize.")
        return

    agent = MultiAgentQLearning(env.n_states, env.n_actions, agent_count=env.agent_count)

    if os.path.exists(q_table_path):
        with open(q_table_path, "rb") as f:
            loaded_data = pickle.load(f)
            if isinstance(loaded_data, list):
                agent.q_agents = loaded_data
                agent.shared_q = False
            else:
                agent.q_shared = loaded_data
                agent.shared_q = True
            print("Successfully loaded Q-table")
    else:
        print("Q-table file not found. Cannot visualize learned behavior.")
        return

    state = env.reset()

    temp_dir = "temp_frames_vis"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    frames = []
    done = False
    step = 0
    max_steps = 60

    print("Generating animation...")
    while not done and step < max_steps:
        frame_path = os.path.join(temp_dir, f"step_{step:03d}.png")
        env.render(path=frame_path)
        frames.append(imageio.imread(frame_path))

        state_idx = env.encode_state(state)
        actions = agent.select_greedy_actions(state_idx)

        joint_action = 0
        for idx, act in enumerate(actions):
            joint_action += act * (env.n_actions**idx)

        result = env.step(joint_action)
        state = result.state
        done = result.done
        step += 1

    gif_name = f"{prefix_part}demo.gif" if prefix else "final_demo.gif"
    gif_path = os.path.join(gif_dir, gif_name)
    imageio.mimsave(gif_path, frames, duration=0.7)
    print(f"GIF saved at: {gif_path}")
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained Q-learning policy")
    parser.add_argument(
        "--prefix",
        default="",
        help="Artifact prefix, e.g. 'nature' or 'adversary' for comparison runs",
    )
    args = parser.parse_args()
    run_visualization(prefix=args.prefix)
