"""Tabular Q-learning agents.

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


class MultiAgentQLearning:
    """Two-agent Q-learning with optional shared Q-table."""

    DEFAULT_ALPHA = 0.1
    DEFAULT_GAMMA = 0.95
    DEFAULT_EPSILON = 0.1
    AGENT_COUNT = 2

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        agent_count: int = 2,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        epsilon: float = DEFAULT_EPSILON,
        shared_q: bool = True,
    ) -> None:
        # Joint action space: action0 x action1 x ...
        self.n_actions = n_actions
        self.agent_count = agent_count
        self.n_joint_actions = n_actions**agent_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.shared_q = shared_q

        if shared_q:
            self.q_shared = np.zeros((n_states, self.n_joint_actions), dtype=float)
            self.q_agents = None
        else:
            self.q_shared = None
            self.q_agents = [
                np.zeros((n_states, self.n_actions), dtype=float) for _ in range(self.agent_count)
            ]

    def select_actions(self, state_idx: int) -> tuple[int, ...]:
        """Select actions for both agents (epsilon-greedy)."""
        if self.shared_q:
            if random.random() < self.epsilon:
                joint = random.randrange(self.n_joint_actions)
            else:
                joint = int(np.argmax(self.q_shared[state_idx]))
            return tuple(self._decode_joint_action(joint))

        # Independent action selection when not sharing a Q-table.
        actions = []
        for agent_idx in range(self.agent_count):
            if random.random() < self.epsilon:
                actions.append(random.randrange(self.n_actions))
            else:
                actions.append(int(np.argmax(self.q_agents[agent_idx][state_idx])))
        return tuple(actions)

    def select_greedy_actions(self, state_idx: int) -> tuple[int, ...]:
        """Select the best-known actions without exploration."""
        if self.shared_q:
            joint = int(np.argmax(self.q_shared[state_idx]))
            return tuple(self._decode_joint_action(joint))

        actions = []
        for agent_idx in range(self.agent_count):
            actions.append(int(np.argmax(self.q_agents[agent_idx][state_idx])))
        return tuple(actions)

    def update(
        self,
        state_idx: int,
        actions: tuple[int, int],
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> None:
        """Update Q-values for either shared or separate tables."""
        if self.shared_q:
            joint = self._encode_joint_action(actions)
            best_next = 0.0 if done else float(np.max(self.q_shared[next_state_idx]))
            td_target = reward + self.gamma * best_next
            td_error = td_target - self.q_shared[state_idx, joint]
            self.q_shared[state_idx, joint] += self.alpha * td_error
            return

        for agent_idx in range(self.agent_count):
            action = actions[agent_idx]
            best_next = 0.0 if done else float(np.max(self.q_agents[agent_idx][next_state_idx]))
            td_target = reward + self.gamma * best_next
            td_error = td_target - self.q_agents[agent_idx][state_idx, action]
            self.q_agents[agent_idx][state_idx, action] += self.alpha * td_error

    def _encode_joint_action(self, actions: tuple[int, ...]) -> int:
        """Encode per-agent actions into a single joint action index."""
        joint = 0
        for idx, action in enumerate(actions):
            joint += action * (self.n_actions**idx)
        return joint

    def _decode_joint_action(self, joint: int) -> list[int]:
        """Decode a joint action index into per-agent actions."""
        actions: list[int] = []
        remaining = joint
        for _ in range(self.agent_count):
            actions.append(remaining % self.n_actions)
            remaining //= self.n_actions
        return actions
