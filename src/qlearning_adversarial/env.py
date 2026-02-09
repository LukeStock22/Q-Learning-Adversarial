"""Gridworld environment and adversary dynamics.

This is a minimal gridworld: an agent moves on an N x N grid trying to reach
the goal. Each move has a small cost, and reaching the goal gives a reward.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepResult:
    """Return type for one environment step."""
    state: tuple[int, int]
    reward: float
    done: bool
    info: dict


class GridworldEnv:
    def __init__(
        self,
        size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
        obstacles: set[tuple[int, int]] | None = None,
        max_steps: int = 200,
    ) -> None:
        # Grid size and start/goal positions.
        self.size = size
        self.start = start
        self.goal = goal if goal is not None else (size - 1, size - 1)
        # Optional blocked cells (agent cannot enter these).
        self.obstacles = obstacles or set()
        # Episode length limit.
        self.max_steps = max_steps
        # Actions: 0=up, 1=right, 2=down, 3=left.
        self.n_actions = 4
        # Each grid cell is a distinct state.
        self.n_states = size * size
        self._steps = 0
        self._state = start

    def reset(self) -> tuple[int, int]:
        """Start a new episode and return the initial state."""
        self._steps = 0
        self._state = self.start
        return self._state

    def encode_state(self, state: tuple[int, int]) -> int:
        """Map a 2D (row, col) state to a single integer index."""
        row, col = state
        return row * self.size + col

    def step(self, action: int) -> StepResult:
        """Apply an action and return (next_state, reward, done, info)."""
        self._steps += 1
        row, col = self._state

        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)

        proposed = (row, col)
        # Default step cost encourages shorter paths.
        reward = -1.0
        if proposed in self.obstacles:
            # Collision: stay in place and apply a larger penalty.
            proposed = self._state
            reward = -2.0

        self._state = proposed
        done = self._state == self.goal or self._steps >= self.max_steps
        if self._state == self.goal:
            # Reaching the goal gives a positive reward.
            reward = 10.0

        return StepResult(self._state, reward, done, {})
