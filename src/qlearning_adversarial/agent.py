"""Tabular Q-learning agent.

Q-learning learns a table of values Q(s, a) that estimates how good it is
to take action a in state s. "Good" means expected total reward.
"""

from __future__ import annotations

import random

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        # Hyperparameters:
        # - alpha: learning rate (how fast we change Q-values)
        # - gamma: discount factor (how much future reward matters)
        # - epsilon: exploration probability (random action rate)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table initialized to zeros.
        self.q = np.zeros((n_states, n_actions), dtype=float)

    def select_action(self, state_idx: int) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            # Explore: choose a random action.
            return random.randrange(self.n_actions)
        # Exploit: choose the best known action.
        return int(np.argmax(self.q[state_idx]))

    def select_greedy_action(self, state_idx: int) -> int:
        """Select the best-known action without exploration (for evaluation)."""
        return int(np.argmax(self.q[state_idx]))

    def update(self, state_idx: int, action: int, reward: float, next_state_idx: int, done: bool) -> None:
        """Update Q(s,a) using the standard Q-learning rule."""
        # If episode ended, there is no future value.
        best_next = 0.0 if done else float(np.max(self.q[next_state_idx]))
        # TD target = immediate reward + discounted best future value.
        td_target = reward + self.gamma * best_next
        # TD error = target - current estimate.
        td_error = td_target - self.q[state_idx, action]
        # Move the estimate toward the target.
        self.q[state_idx, action] += self.alpha * td_error
