# Q-Learning-Adversarial

Tabular Q-Learning in a grid-based warehouse environment with stochastic disturbances and strategic adversaries.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project Goals
- Compare stochastic vs adversarial training for robustness in dynamic gridworlds
- Evaluate generalization to unseen layouts and disruption patterns

## Repo Status
This is the initial skeleton. Core environment + training loop to be added.

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
- `src/qlearning_adversarial/env.py`: Gridworld environment and adversary dynamics (placeholder).
- `src/qlearning_adversarial/agent.py`: Tabular Q-learning agent (placeholder).
- `src/qlearning_adversarial/train.py`: Training loop (placeholder).
- `src/qlearning_adversarial/eval.py`: Evaluation utilities (placeholder).
- `src/qlearning_adversarial/utils.py`: Shared helpers (placeholder).
- `tests/`: Automated tests.
- `tests/test_smoke.py`: Basic smoke test placeholder.
