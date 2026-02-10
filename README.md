# Q-Learning-Adversarial

Tabular Q-Learning in a grid-based warehouse environment with stochastic disturbances and strategic adversaries.

Current focus: a two-agent gridworld with multiple packages, fixed shelves, and per-episode spill obstacles (no adversaries yet).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run
```bash
PYTHONPATH=src python -m qlearning_adversarial.main
```

## Configuration
Edit `configs/default.yaml` to change grid size, training hyperparameters, evaluation episodes, and output paths.

## Evaluation Modes
- ID-A: same exact layout as training (same shelf + package/destination), but spills randomized each episode.
- OOD: new shelf layout and new package/destination placements; same disturbance parameters.

## Outputs
Running the program writes:
- `outputs/rewards.csv`: per-episode reward and moving average
- `outputs/learning_curve.png`: simple training curve plot
- `outputs/gridworld_layout.png`: static image of the grid layout (starts, packages, destinations, shelf)
- `outputs/metrics.txt`: ID-A and OOD evaluation averages

## Project Goals
- Compare stochastic vs adversarial training for robustness in dynamic gridworlds
- Evaluate generalization to unseen layouts and disruption patterns

## Repo Status
Working multi-agent skeleton with training, evaluation, and plotting.

## Customizability
The following parameters can be edited in `configs/default.yaml`:
- `project.grid_size`: grid width/height
- `project.seed`: base RNG seed
- `project.output_dir`: output directory
- `run.tier`: `debug` (fast) or `report` (longer)
- `run.debug.*` / `run.report.*`: train and eval episode counts per tier
- `training.max_steps`: max steps per episode
- `training.alpha`, `training.gamma`, `training.epsilon`: Q-learning hyperparameters
- `training.agent_count`: number of agents (1 or 2)
- `training.num_packages`: number of packages/destinations
- `training.shared_q`: shared Q-table toggle for two agents
- `training.spill_count`: number of randomized spill obstacles per episode
- `eval.mode`: `id_a`, `ood`, or `both`
- `eval.max_steps`: max steps per evaluation episode

## Repository Structure
- `.git/`: Git metadata for version control.
- `.gitignore`: Ignore rules for local artifacts (envs, caches, outputs).
- `.venv/`: Local Python virtual environment (not committed).
- `PROJECT_DESCRIPTION.md`: Project brief, literature context, and timeline.
- `README.md`: Project overview and setup instructions.
- `requirements.txt`: Python dependencies for the baseline setup.
- `configs/`: Configuration files for experiments.
- `configs/default.yaml`: Default training and environment settings.
- `data/`: Location for raw or generated datasets.
- `data/README.md`: Notes on data storage and versioning guidance.
- `notebooks/`: Exploratory analysis and quick experiments.
- `notebooks/README.md`: Notes on notebook usage.
- `outputs/`: Training outputs, plots, and checkpoints.
- `outputs/.gitkeep`: Keeps the outputs directory tracked in git.
- `scripts/`: Helper scripts for running workflows.
- `scripts/run_train.sh`: Shell script to run the entrypoint.
- `src/`: Python source package root.
- `src/qlearning_adversarial/`: Core package for the project.
- `src/qlearning_adversarial/__init__.py`: Package initializer.
- `src/qlearning_adversarial/main.py`: Entry point for running training/evaluation.
- `src/qlearning_adversarial/env.py`: Two-agent gridworld with multi-package pickup/delivery, collisions, and fixed shelf.
- `src/qlearning_adversarial/agent.py`: Single- and multi-agent Q-learning implementations with shared-table option.
- `src/qlearning_adversarial/train.py`: Training loop and reward smoothing.
- `src/qlearning_adversarial/eval.py`: Greedy evaluation loop.
- `src/qlearning_adversarial/utils.py`: Seeding and filesystem helpers.
- `tests/`: Automated tests.
- `tests/test_smoke.py`: Basic smoke test placeholder.
