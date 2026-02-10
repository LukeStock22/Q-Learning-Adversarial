"""Gridworld environment and adversary dynamics.

Two agents navigate an N x N grid to pick up multiple packages and deliver them
to their unique destinations. Each move has a small cost, collisions and
obstacles are penalized, and deliveries are rewarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import random

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class StepResult:
    """Return type for one environment step."""
    state: tuple[tuple[int, int], ...]
    reward: float
    done: bool
    info: dict


class GridworldEnv:
    # Action indices.
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_COUNT = 4

    # Package state encoding.
    PACKAGE_AT_PICKUP = 0
    PACKAGE_WITH_AGENT0 = 1
    PACKAGE_WITH_AGENT1 = 2
    PACKAGE_DELIVERED = 3
    PACKAGE_STATE_BASE = 4

    # Rewards and penalties.
    STEP_PENALTY = -2.0
    OBSTACLE_PENALTY = -3.0
    COLLISION_PENALTY = -5.0
    PICKUP_REWARD = 2.0
    DELIVERY_REWARD = 15.0
    DEFAULT_SPILL_COUNT = 2

    # Render layer values.
    RENDER_EMPTY = 0
    RENDER_START0 = 1
    RENDER_START1 = 2
    RENDER_PACKAGE = 3
    RENDER_DESTINATION = 4
    RENDER_SHELF = 5
    RENDER_SPILL = 6

    def __init__(
        self,
        size: int = 10,
        starts: tuple[tuple[int, int], ...] | None = None,
        package_locations: list[tuple[int, int]] | None = None,
        destinations: list[tuple[int, int]] | None = None,
        num_packages: int = 3,
        obstacles: set[tuple[int, int]] | None = None,
        max_steps: int = 200,
        spill_count: int = DEFAULT_SPILL_COUNT,
        agent_count: int = 2,
    ) -> None:
        # Grid size and agent start positions.
        self.size = size
        self.agent_count = agent_count
        default_starts = ((0, 0), (size - 1, 0))
        if starts is not None:
            self.starts = starts
        else:
            self.starts = default_starts[:agent_count]
        # Package pickup + dropoff locations.
        self.num_packages = num_packages
        self.package_locations = package_locations or []
        self.destinations = destinations or []
        # Fixed "shelf" obstacles (2x1) per run.
        self.shelf_obstacles = obstacles or set()
        if not self.shelf_obstacles:
            self.shelf_obstacles = self._place_fixed_shelf()
        # Dynamic spill obstacles per episode.
        self.spill_count = spill_count
        self.spill_obstacles: set[tuple[int, int]] = set()
        # Episode length limit.
        self.max_steps = max_steps
        # Actions: 0=up, 1=right, 2=down, 3=left.
        self.n_actions = self.ACTION_COUNT
        # Each grid cell for two agents + package state (4^num_packages).
        self.n_states = (size * size) ** 2 * (self.PACKAGE_STATE_BASE ** self.num_packages)
        self._steps = 0
        self._agent_positions = list(self.starts)
        self._package_state: list[int] = []
        self._agent_carrying: list[int | None] = []

        # If package/destination positions are not provided, generate them.
        if not self.package_locations or not self.destinations:
            self._generate_packages_and_destinations()

    def reset(self) -> tuple[tuple[int, int], ...]:
        """Start a new episode and return the initial state."""
        self._steps = 0
        self._agent_positions = list(self.starts)
        # Package state codes are defined by PACKAGE_* constants.
        self._package_state = [self.PACKAGE_AT_PICKUP for _ in range(self.num_packages)]
        self._agent_carrying = [None for _ in range(self.agent_count)]
        self.spill_obstacles = self._sample_spills()
        return tuple(self._agent_positions)

    def encode_state(self, state: tuple[tuple[int, int], ...]) -> int:
        """Map (pos_agents..., package_state) to a single index."""
        positions = [r * self.size + c for (r, c) in state]
        package_code = 0
        for idx, value in enumerate(self._package_state):
            package_code += value * (self.PACKAGE_STATE_BASE**idx)
        pos_base = self.size * self.size
        pos_code = 0
        for idx, pos in enumerate(positions):
            pos_code += pos * (pos_base**idx)
        return pos_code + package_code * (pos_base**self.agent_count)

    def step(self, action: int) -> StepResult:
        """Apply an action and return (next_state, reward, done, info)."""
        self._steps += 1
        actions = self._decode_joint_action(action)
        positions = list(self._agent_positions)
        next_positions = [self._move(pos, act) for pos, act in zip(positions, actions)]

        reward = self.STEP_PENALTY  # step cost for two agents

        # Obstacle handling: blocked agents stay in place and get a penalty.
        obstacles = self._all_obstacles()
        for idx, (pos, nxt) in enumerate(zip(positions, next_positions)):
            if nxt in obstacles:
                next_positions[idx] = pos
                reward += self.OBSTACLE_PENALTY

        # Collision handling: same cell or swapping positions (2-agent case).
        if self.agent_count > 1:
            if next_positions[0] == next_positions[1] or (
                next_positions[0] == positions[1] and next_positions[1] == positions[0]
            ):
                next_positions = positions
                reward += self.COLLISION_PENALTY

        self._agent_positions = next_positions

        # Auto-pickup: if an agent is on a package and not carrying one.
        reward += self._handle_pickups()
        # Delivery: if an agent carrying a package reaches its destination.
        reward += self._handle_deliveries()

        done = self._steps >= self.max_steps or all(
            state == self.PACKAGE_DELIVERED for state in self._package_state
        )
        state = tuple(self._agent_positions)
        return StepResult(state, reward, done, {})

    def _move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        """Apply one action to a single agent position."""
        row, col = pos
        if action == self.ACTION_UP:  # up
            row = max(0, row - 1)
        elif action == self.ACTION_RIGHT:  # right
            col = min(self.size - 1, col + 1)
        elif action == self.ACTION_DOWN:  # down
            row = min(self.size - 1, row + 1)
        elif action == self.ACTION_LEFT:  # left
            col = max(0, col - 1)
        return (row, col)

    def _handle_pickups(self) -> float:
        """Pick up packages if an agent stands on them and is free to carry."""
        bonus = 0.0
        for agent_idx, pos in enumerate(self._agent_positions):
            if self._agent_carrying[agent_idx] is not None:
                continue
            for pkg_idx, pkg_pos in enumerate(self.package_locations):
                if self._package_state[pkg_idx] == self.PACKAGE_AT_PICKUP and pos == pkg_pos:
                    self._package_state[pkg_idx] = (
                        self.PACKAGE_WITH_AGENT0
                        if agent_idx == 0
                        else self.PACKAGE_WITH_AGENT1
                    )
                    self._agent_carrying[agent_idx] = pkg_idx
                    bonus += self.PICKUP_REWARD
                    break
        return bonus

    def _handle_deliveries(self) -> float:
        """Deliver packages if an agent reaches its destination."""
        bonus = 0.0
        for agent_idx, pos in enumerate(self._agent_positions):
            pkg_idx = self._agent_carrying[agent_idx]
            if pkg_idx is None:
                continue
            if pos == self.destinations[pkg_idx]:
                self._package_state[pkg_idx] = self.PACKAGE_DELIVERED
                self._agent_carrying[agent_idx] = None
                bonus += self.DELIVERY_REWARD
        return bonus

    def _generate_packages_and_destinations(self) -> None:
        """Generate package pickup + destination locations without overlap."""
        occupied = set(self.starts) | set(self.shelf_obstacles)
        self.package_locations = []
        self.destinations = []

        def sample_cell() -> tuple[int, int]:
            while True:
                cell = (random.randrange(self.size), random.randrange(self.size))
                if cell not in occupied:
                    return cell

        for _ in range(self.num_packages):
            pkg = sample_cell()
            occupied.add(pkg)
            dest = sample_cell()
            occupied.add(dest)
            self.package_locations.append(pkg)
            self.destinations.append(dest)

    def _place_fixed_shelf(self) -> set[tuple[int, int]]:
        """Create a fixed 2x1 obstacle (horizontal or vertical) for this run."""
        while True:
            horizontal = random.random() < 0.5
            if horizontal:
                row = random.randrange(self.size)
                col = random.randrange(self.size - 1)
                cells = {(row, col), (row, col + 1)}
            else:
                row = random.randrange(self.size - 1)
                col = random.randrange(self.size)
                cells = {(row, col), (row + 1, col)}
            if cells.isdisjoint(self.starts):
                return cells

    def _sample_spills(self) -> set[tuple[int, int]]:
        """Generate per-episode spill obstacles (1x1 cells)."""
        spills: set[tuple[int, int]] = set()
        occupied = (
            set(self.starts)
            | set(self.shelf_obstacles)
            | set(self.package_locations)
            | set(self.destinations)
        )
        while len(spills) < self.spill_count:
            cell = (random.randrange(self.size), random.randrange(self.size))
            if cell not in occupied:
                spills.add(cell)
                occupied.add(cell)
        return spills

    def _decode_joint_action(self, action: int) -> list[int]:
        """Decode a joint action into per-agent actions."""
        actions: list[int] = []
        remaining = action
        for _ in range(self.agent_count):
            actions.append(remaining % self.n_actions)
            remaining //= self.n_actions
        return actions

    def _all_obstacles(self) -> set[tuple[int, int]]:
        """Return the union of fixed shelves and per-episode spills."""
        return self.shelf_obstacles | self.spill_obstacles

    def render(self, path: str | None = None) -> None:
        """Render a static view of the grid with starts, packages, and obstacles."""
        # Build a simple integer grid:
        # 0=empty, 1=start0, 2=start1, 3=package, 4=destination, 5=shelf, 6=spill.
        grid = np.zeros((self.size, self.size), dtype=int)
        for (r, c) in self.shelf_obstacles:
            grid[r, c] = self.RENDER_SHELF
        for (r, c) in self.spill_obstacles:
            grid[r, c] = self.RENDER_SPILL
        grid[self.starts[0]] = self.RENDER_START0
        if self.agent_count > 1:
            grid[self.starts[1]] = self.RENDER_START1
        for pkg in self.package_locations:
            grid[pkg] = self.RENDER_PACKAGE
        for dest in self.destinations:
            grid[dest] = self.RENDER_DESTINATION

        # Color map for the grid.
        colors = np.array(
            [
                [1.0, 1.0, 1.0],  # empty - white
                [0.2, 0.6, 1.0],  # start0 - blue
                [0.2, 0.4, 0.8],  # start1 - darker blue
                [1.0, 0.8, 0.2],  # package - orange
                [0.2, 0.8, 0.2],  # destination - green
                [0.3, 0.3, 0.3],  # shelf - dark gray
                [0.7, 0.2, 0.2],  # spill - red
            ]
        )

        image = colors[grid]
        plt.figure(figsize=(4.5, 4.5))
        plt.imshow(image, interpolation="none")
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.grid(which="both", color="black", linewidth=0.5)
        plt.title("Gridworld layout")

        handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[1], markersize=10, label="Start A0"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[3], markersize=10, label="Package"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[4], markersize=10, label="Destination"),
        ]
        if self.agent_count > 1:
            handles.insert(
                1,
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=colors[2],
                    markersize=10,
                    label="Start A1",
                ),
            )
        if self.shelf_obstacles:
            handles.append(
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[5], markersize=10, label="Shelf")
            )
        if self.spill_obstacles:
            handles.append(
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[6], markersize=10, label="Spill")
            )

        # Overlay current agent positions (circles) for clarity.
        a0 = self._agent_positions[0]
        plt.scatter([a0[1]], [a0[0]], c="black", marker="o", s=60)
        if self.agent_count > 1:
            a1 = self._agent_positions[1]
            plt.scatter([a1[1]], [a1[0]], c="black", marker="x", s=60)
        plt.legend(handles=handles, loc="upper right")
        plt.tight_layout()

        if path:
            plt.savefig(path)
        plt.close()
