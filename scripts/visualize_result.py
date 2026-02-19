import pickle
import imageio.v2 as imageio
import os
import shutil
from qlearning_adversarial.env import GridworldEnv
from qlearning_adversarial.agent import MultiAgentQLearning

def run_visualization():
    output_dir = "outputs"
    config_path = os.path.join(output_dir, "env_config.pkl")
    q_table_path = os.path.join(output_dir, "q_table.pkl")


    if os.path.exists(config_path):
        print(f"Loading saved map configuration: {config_path}")
        with open(config_path, "rb") as f:
            env_config = pickle.load(f)
 
        env = GridworldEnv(**env_config)
    else:
        print("Map configuration file not found, a new random map will be generated (which may not be consistent with the one used during training)")
        env = GridworldEnv(size=10, agent_count=2, spill_count=2)

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
            print("successfully loaded Q-Table")
    else:
        print("without Q-Table file! Cannot visualize learned behavior.")
        return


    state = env.reset()
    
    temp_dir = "temp_frames_vis"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    frames = []
    done = False
    step = 0
    max_steps = 50

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

    imageio.mimsave(os.path.join(output_dir, "final_demo.gif"), frames, duration=1.0)
    print(f"gif saved at: {os.path.join(output_dir, 'final_demo.gif')}")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    run_visualization()