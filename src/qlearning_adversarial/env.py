"""Gridworld environment with configurable disturbance strategies.

Scenarios:
- nature: static shelves + random forklifts (non-strategic movers)
- adversary: static shelves + pursuit adversary (strategic jammer)
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
    # Scenario identifiers.
    SCENARIO_NATURE = "nature"
    SCENARIO_ADVERSARY = "adversary"
    ADVERSARY_POLICY_DETERMINISTIC = "deterministic"
    ADVERSARY_POLICY_LEARNING = "learning"
    ADVERSARY_OBJECTIVE_HEURISTIC = "heuristic"
    ADVERSARY_OBJECTIVE_ZERO_SUM = "zero_sum"

    # Action indices.
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_COUNT = 4
    ADVERSARY_ACTION_STAY = 4
    ADVERSARY_ACTION_COUNT = 5

    # Package state encoding.
    PACKAGE_AT_PICKUP = 0
    PACKAGE_WITH_AGENT0 = 1
    PACKAGE_WITH_AGENT1 = 2
    PACKAGE_DELIVERED = 3
    PACKAGE_STATE_BASE = 4

    # Rewards and penalties.
    DEFAULT_STEP_PENALTY = -2.0
    DEFAULT_OBSTACLE_PENALTY = -3.0
    DEFAULT_COLLISION_PENALTY = -5.0
    DEFAULT_FORKLIFT_PENALTY = -50.0
    DEFAULT_ADVERSARY_PENALTY = -50.0
    DEFAULT_PICKUP_REWARD = 10.0
    DEFAULT_DELIVERY_REWARD = 80.0
    DEFAULT_DISTANCE_SHAPING_ENABLED = False
    DEFAULT_DISTANCE_SHAPING_SCALE = 0.0
    DEFAULT_INCLUDE_RELATIVE_PACKAGE_DESTINATION = False

    # Defaults.
    DEFAULT_SPILL_COUNT = 0
    DEFAULT_FORKLIFT_COUNT = 1
    DEFAULT_FORKLIFT_MOVE_PROB = 0.5
    DEFAULT_ADVERSARY_RANDOM_TIEBREAK = True
    DEFAULT_ADVERSARY_POLICY = ADVERSARY_POLICY_DETERMINISTIC
    DEFAULT_ADVERSARY_MOVE_PROB = 0.5
    DEFAULT_ADVERSARY_MAX_MOVES = 1000
    DEFAULT_ADVERSARY_LEARNING_ALPHA = 0.2
    DEFAULT_ADVERSARY_LEARNING_GAMMA = 0.95
    DEFAULT_ADVERSARY_LEARNING_EPSILON_START = 0.2
    DEFAULT_ADVERSARY_LEARNING_EPSILON_END = 0.05
    DEFAULT_ADVERSARY_LEARNING_EPSILON_DECAY_EPISODES = 5000
    DEFAULT_ADVERSARY_LEARNING_PROGRESS_REWARD_SCALE = 1.0
    DEFAULT_ADVERSARY_LEARNING_CATCH_REWARD = 25.0
    DEFAULT_ADVERSARY_LEARNING_OBJECTIVE = ADVERSARY_OBJECTIVE_HEURISTIC
    DEFAULT_ADVERSARY_FREEZE_EPISODE = 0  # 0 = never freeze
    DEFAULT_ADVERSARY_PROXIMITY_RADIUS = 0  # 0 = full position; >0 = 5-bucket directional proximity
    DEFAULT_INCLUDE_COARSE_DESTINATION_DIRECTION = False  # octant direction to pkg/dest (9×9=81 values)

    # Render layer values.
    RENDER_EMPTY = 0
    RENDER_START0 = 1
    RENDER_START1 = 2
    RENDER_PACKAGE = 3
    RENDER_DESTINATION = 4
    RENDER_SHELF = 5
    RENDER_SPILL = 6
    RENDER_ADVERSARY = 7
    RENDER_FORKLIFT = 8

    def __init__(
        self,
        size: int = 10,
        starts: tuple[tuple[int, int], ...] | None = None,
        package_locations: list[tuple[int, int]] | None = None,
        destinations: list[tuple[int, int]] | None = None,
        forklift_starts: list[tuple[int, int]] | None = None,
        adversary_start: tuple[int, int] | None = None,
        num_packages: int = 3,
        obstacles: set[tuple[int, int]] | None = None,
        max_steps: int = 200,
        spill_count: int = DEFAULT_SPILL_COUNT,
        agent_count: int = 1,
        scenario: str = SCENARIO_NATURE,
        forklift_count: int = DEFAULT_FORKLIFT_COUNT,
        forklift_move_prob: float = DEFAULT_FORKLIFT_MOVE_PROB,
        adversary_random_tiebreak: bool = DEFAULT_ADVERSARY_RANDOM_TIEBREAK,
        adversary_policy: str = DEFAULT_ADVERSARY_POLICY,
        adversary_move_prob: float = DEFAULT_ADVERSARY_MOVE_PROB,
        adversary_max_moves: int = DEFAULT_ADVERSARY_MAX_MOVES,
        adversary_learning_alpha: float = DEFAULT_ADVERSARY_LEARNING_ALPHA,
        adversary_learning_gamma: float = DEFAULT_ADVERSARY_LEARNING_GAMMA,
        adversary_learning_epsilon_start: float = DEFAULT_ADVERSARY_LEARNING_EPSILON_START,
        adversary_learning_epsilon_end: float = DEFAULT_ADVERSARY_LEARNING_EPSILON_END,
        adversary_learning_epsilon_decay_episodes: int = DEFAULT_ADVERSARY_LEARNING_EPSILON_DECAY_EPISODES,
        adversary_learning_progress_reward_scale: float = DEFAULT_ADVERSARY_LEARNING_PROGRESS_REWARD_SCALE,
        adversary_learning_catch_reward: float = DEFAULT_ADVERSARY_LEARNING_CATCH_REWARD,
        adversary_learning_objective: str = DEFAULT_ADVERSARY_LEARNING_OBJECTIVE,
        adversary_freeze_episode: int = DEFAULT_ADVERSARY_FREEZE_EPISODE,
        adversary_enabled: bool | None = None,
        step_penalty: float = DEFAULT_STEP_PENALTY,
        obstacle_penalty: float = DEFAULT_OBSTACLE_PENALTY,
        collision_penalty: float = DEFAULT_COLLISION_PENALTY,
        forklift_penalty: float = DEFAULT_FORKLIFT_PENALTY,
        adversary_penalty: float = DEFAULT_ADVERSARY_PENALTY,
        pickup_reward: float = DEFAULT_PICKUP_REWARD,
        delivery_reward: float = DEFAULT_DELIVERY_REWARD,
        distance_shaping_enabled: bool = DEFAULT_DISTANCE_SHAPING_ENABLED,
        distance_shaping_scale: float = DEFAULT_DISTANCE_SHAPING_SCALE,
        include_relative_package_destination: bool = DEFAULT_INCLUDE_RELATIVE_PACKAGE_DESTINATION,
        adversary_proximity_radius: int = DEFAULT_ADVERSARY_PROXIMITY_RADIUS,
        include_coarse_destination_direction: bool = DEFAULT_INCLUDE_COARSE_DESTINATION_DIRECTION,
        shelf_count: int = 1,
        adversary_count: int = 1,
    ) -> None:
        self.size = size
        self.agent_count = agent_count
        self.scenario = scenario
        if self.scenario not in (self.SCENARIO_NATURE, self.SCENARIO_ADVERSARY):
            raise ValueError(f"Unknown scenario: {self.scenario}")
        if adversary_policy not in (self.ADVERSARY_POLICY_DETERMINISTIC, self.ADVERSARY_POLICY_LEARNING):
            raise ValueError(f"Unknown adversary policy: {adversary_policy}")
        if adversary_learning_objective not in (
            self.ADVERSARY_OBJECTIVE_HEURISTIC,
            self.ADVERSARY_OBJECTIVE_ZERO_SUM,
        ):
            raise ValueError(f"Unknown adversary learning objective: {adversary_learning_objective}")

        self.adversary_count = max(1, int(adversary_count))
        if self.adversary_count > 1 and adversary_policy == self.ADVERSARY_POLICY_LEARNING:
            raise ValueError("adversary_count > 1 is only supported with deterministic adversary policy.")

        if adversary_enabled is None:
            self.adversary_enabled = self.scenario == self.SCENARIO_ADVERSARY
        else:
            self.adversary_enabled = adversary_enabled

        default_starts = ((0, 0), (size - 1, 0))
        self.starts = starts if starts is not None else default_starts[:agent_count]

        self.num_packages = num_packages
        self.package_locations = package_locations or []
        self.destinations = destinations or []
        self.fixed_forklift_starts = list(forklift_starts) if forklift_starts is not None else None
        self.fixed_adversary_start = adversary_start

        self.shelf_count = max(1, int(shelf_count))
        self.shelf_obstacles = obstacles or set()
        if not self.shelf_obstacles:
            for _ in range(self.shelf_count):
                self.shelf_obstacles |= self._place_fixed_shelf()

        self.spill_count = spill_count
        self.spill_obstacles: set[tuple[int, int]] = set()

        self.forklift_count = forklift_count if self.scenario == self.SCENARIO_NATURE else 0
        self.forklift_move_prob = forklift_move_prob
        self.forklift_positions: list[tuple[int, int]] = []
        self.adversary_random_tiebreak = adversary_random_tiebreak
        self.adversary_policy = adversary_policy
        self.adversary_move_prob = adversary_move_prob
        self.adversary_max_moves = max(0, int(adversary_max_moves))
        self.adversary_learning_alpha = adversary_learning_alpha
        self.adversary_learning_gamma = adversary_learning_gamma
        self.adversary_learning_epsilon_start = adversary_learning_epsilon_start
        self.adversary_learning_epsilon_end = adversary_learning_epsilon_end
        self.adversary_learning_epsilon_decay_episodes = max(1, adversary_learning_epsilon_decay_episodes)
        self.adversary_learning_progress_reward_scale = adversary_learning_progress_reward_scale
        self.adversary_learning_catch_reward = adversary_learning_catch_reward
        self.adversary_learning_objective = adversary_learning_objective
        self.adversary_freeze_episode = max(0, int(adversary_freeze_episode))
        self._adversary_learning_epsilon = adversary_learning_epsilon_start

        # Reward/cost model for tuning.
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty
        self.collision_penalty = collision_penalty
        self.forklift_penalty = forklift_penalty
        self.adversary_penalty = adversary_penalty
        self.pickup_reward = pickup_reward
        self.delivery_reward = delivery_reward
        self.distance_shaping_enabled = distance_shaping_enabled
        self.distance_shaping_scale = distance_shaping_scale
        self.include_relative_package_destination = include_relative_package_destination
        self.adversary_proximity_radius = max(0, int(adversary_proximity_radius))
        self.include_coarse_destination_direction = include_coarse_destination_direction

        self._adversary_positions: list[tuple[int, int]] = []
        self.max_steps = max_steps
        self.n_actions = self.ACTION_COUNT

        pos_base = size * size
        dynamic_slots = self._dynamic_slot_count()
        dynamic_slot_size = 5 if self.adversary_proximity_radius > 0 else pos_base
        self.n_states = (
            (pos_base**self.agent_count)
            * (dynamic_slot_size**dynamic_slots)
            * (self.PACKAGE_STATE_BASE**self.num_packages)
            * ((2 * self.size - 1) ** (4 * self.num_packages) if self.include_relative_package_destination else 1)
            * ((9 ** (2 * self.num_packages)) if self.include_coarse_destination_direction else 1)
        )

        self._steps = 0
        self._agent_positions = list(self.starts)
        self._package_state: list[int] = []
        self._agent_carrying: list[int | None] = []
        self._terminal_status: str | None = None
        self._episode_count = 0
        self._adversary_moves_used = 0
        self._adversary_q_table = np.zeros((size * size * size * size, self.ADVERSARY_ACTION_COUNT), dtype=float)

        if not self.package_locations or not self.destinations:
            self._generate_packages_and_destinations()

    @property
    def adversary_pos(self) -> tuple[int, int] | None:
        """Backward-compatible single adversary position (first adversary, or None)."""
        return self._adversary_positions[0] if self._adversary_positions else None

    @adversary_pos.setter
    def adversary_pos(self, value: tuple[int, int] | None) -> None:
        if value is None:
            self._adversary_positions = []
        elif self._adversary_positions:
            self._adversary_positions[0] = value
        else:
            self._adversary_positions = [value]

    def reset(self) -> tuple[tuple[int, int], ...]:
        """Start a new episode and return the initial state."""
        self._episode_count += 1
        self._steps = 0
        self._agent_positions = list(self.starts)
        self._package_state = [self.PACKAGE_AT_PICKUP for _ in range(self.num_packages)]
        self._agent_carrying = [None for _ in range(self.agent_count)]
        self._terminal_status = None
        self._adversary_moves_used = 0
        self.spill_obstacles = self._sample_spills()

        if self.adversary_enabled:
            self._adversary_positions = self._spawn_adversaries()
        else:
            self._adversary_positions = []

        if self.adversary_policy == self.ADVERSARY_POLICY_LEARNING:
            frozen = self.adversary_freeze_episode > 0 and self._episode_count > self.adversary_freeze_episode
            if frozen:
                self._adversary_learning_epsilon = self.adversary_learning_epsilon_end
            else:
                progress = min(1.0, self._episode_count / self.adversary_learning_epsilon_decay_episodes)
                self._adversary_learning_epsilon = (
                    self.adversary_learning_epsilon_start
                    + progress * (self.adversary_learning_epsilon_end - self.adversary_learning_epsilon_start)
                )

        if self.scenario == self.SCENARIO_NATURE:
            self.forklift_positions = self._spawn_forklifts()
        else:
            self.forklift_positions = []

        return tuple(self._agent_positions)

    def encode_state(self, state: tuple[tuple[int, int], ...]) -> int:
        """Map (agent positions, dynamic entities, package state) to a single index."""
        pos_base = self.size * self.size

        total_idx = 0
        multiplier = 1

        for row, col in state:
            total_idx += (row * self.size + col) * multiplier
            multiplier *= pos_base

        if self.adversary_proximity_radius > 0:
            agent_pos = state[0]
            for prox_code in self._dynamic_proximity_codes(agent_pos):
                total_idx += prox_code * multiplier
                multiplier *= 5
        else:
            for row, col in self._dynamic_positions_for_encoding():
                total_idx += (row * self.size + col) * multiplier
                multiplier *= pos_base

        package_code = 0
        for idx, value in enumerate(self._package_state):
            package_code += value * (self.PACKAGE_STATE_BASE**idx)
        total_idx += package_code * multiplier
        multiplier *= self.PACKAGE_STATE_BASE**self.num_packages

        if self.include_relative_package_destination:
            relative_code = self._encode_relative_package_destination(state)
            total_idx += relative_code * multiplier
            multiplier *= (2 * self.size - 1) ** (4 * self.num_packages)

        if self.include_coarse_destination_direction:
            coarse_code = self._encode_coarse_destination_directions(state)
            total_idx += coarse_code * multiplier

        return total_idx

    def _encode_relative_package_destination(self, state: tuple[tuple[int, int], ...]) -> int:
        """Encode relative package/destination offsets from agent-0 perspective."""
        ref_row, ref_col = state[0]
        offset_base = 2 * self.size - 1
        shift = self.size - 1

        code = 0
        multiplier = 1
        for pkg, dst in zip(self.package_locations, self.destinations):
            deltas = (
                pkg[0] - ref_row,
                pkg[1] - ref_col,
                dst[0] - ref_row,
                dst[1] - ref_col,
            )
            for delta in deltas:
                encoded = delta + shift
                code += encoded * multiplier
                multiplier *= offset_base
        return code

    def step(self, action: int) -> StepResult:
        """Apply an action and return (next_state, reward, done, info)."""
        self._steps += 1
        actions = self._decode_joint_action(action)
        objective_before = self._total_objective_distance()
        adversary_transition = None

        current_positions = list(self._agent_positions)
        next_positions = [self._move(pos, act) for pos, act in zip(current_positions, actions)]

        reward = self.step_penalty

        obstacles = self._all_obstacles()
        for idx, (current, nxt) in enumerate(zip(current_positions, next_positions)):
            if nxt in obstacles:
                next_positions[idx] = current
                reward += self.obstacle_penalty

        # Dynamic hazard update + collision check differs by scenario.
        if self.scenario == self.SCENARIO_NATURE:
            self._move_forklifts()
            if any(pos in self.forklift_positions for pos in next_positions):
                reward += self.forklift_penalty
                self._agent_positions = next_positions
                self._terminal_status = "caught"
                return StepResult(tuple(self._agent_positions), reward, True, {"caught": "forklift"})
        elif self.adversary_enabled and self._adversary_positions:
            # Adversary can move at most once per environment step, with an episode-level move budget.
            can_move_this_episode = self._adversary_moves_used < self.adversary_max_moves
            should_move_now = random.random() <= self.adversary_move_prob
            allow_adversary_move = can_move_this_episode and should_move_now
            if self.adversary_policy == self.ADVERSARY_POLICY_LEARNING:
                # In zero-sum mode, the adversary learns from every step reward.
                adversary_transition = self._move_adversary_learning(
                    next_positions, allow_move=allow_adversary_move
                )
            elif allow_adversary_move:
                self._move_adversary(next_positions)
            if allow_adversary_move:
                self._adversary_moves_used += 1
            if any(pos in set(self._adversary_positions) for pos in next_positions):
                reward += self.adversary_penalty
                self._agent_positions = next_positions
                self._terminal_status = "caught"
                if adversary_transition is not None:
                    self._update_adversary_q(
                        transition=adversary_transition,
                        agent_reward=reward,
                        done=True,
                    )
                return StepResult(tuple(self._agent_positions), reward, True, {"caught": "adversary"})

        if self.agent_count > 1:
            if next_positions[0] == next_positions[1] or (
                next_positions[0] == current_positions[1] and next_positions[1] == current_positions[0]
            ):
                next_positions = current_positions
                reward += self.collision_penalty

        self._agent_positions = next_positions

        reward += self._handle_pickups()
        reward += self._handle_deliveries()
        if self.distance_shaping_enabled:
            objective_after = self._total_objective_distance()
            reward += self.distance_shaping_scale * (objective_before - objective_after)

        done = self._steps >= self.max_steps or all(
            state == self.PACKAGE_DELIVERED for state in self._package_state
        )
        if done and all(state == self.PACKAGE_DELIVERED for state in self._package_state):
            self._terminal_status = "delivered"
        if self.adversary_enabled and self.adversary_policy == self.ADVERSARY_POLICY_LEARNING and adversary_transition is not None:
            self._update_adversary_q(
                transition=adversary_transition,
                agent_reward=reward,
                done=done,
            )
        return StepResult(tuple(self._agent_positions), reward, done, {})

    def _move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        row, col = pos
        if action == self.ACTION_UP:
            row = max(0, row - 1)
        elif action == self.ACTION_RIGHT:
            col = min(self.size - 1, col + 1)
        elif action == self.ACTION_DOWN:
            row = min(self.size - 1, row + 1)
        elif action == self.ACTION_LEFT:
            col = max(0, col - 1)
        return (row, col)

    def _spawn_adversaries(self) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        occupied = self._reserved_cells()
        for i in range(self.adversary_count):
            if i == 0 and self.fixed_adversary_start is not None:
                positions.append(self.fixed_adversary_start)
                occupied = occupied | {self.fixed_adversary_start}
            else:
                while True:
                    cell = (random.randrange(self.size), random.randrange(self.size))
                    if cell not in occupied:
                        positions.append(cell)
                        occupied = occupied | {cell}
                        break
        return positions

    def _move_adversary(self, agent_positions: list[tuple[int, int]]) -> None:
        if self.adversary_policy == self.ADVERSARY_POLICY_LEARNING:
            self._move_adversary_learning(agent_positions)
            return
        self._move_adversary_deterministic(agent_positions)

    def _move_adversary_deterministic(self, agent_positions: list[tuple[int, int]]) -> None:
        for i in range(len(self._adversary_positions)):
            adv_r, adv_c = self._adversary_positions[i]
            target = min(agent_positions, key=lambda pos: abs(pos[0] - adv_r) + abs(pos[1] - adv_c))
            tgt_r, tgt_c = target

            candidates = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = adv_r + dr, adv_c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) not in self._all_obstacles():
                        candidates.append((nr, nc))
            if not candidates:
                continue

            distances = [abs(pos[0] - tgt_r) + abs(pos[1] - tgt_c) for pos in candidates]
            best_distance = min(distances)
            best_candidates = [pos for pos, dist in zip(candidates, distances) if dist == best_distance]
            # Always random tie-break among equally good moves.
            self._adversary_positions[i] = random.choice(best_candidates)

    def _move_adversary_learning(
        self,
        agent_positions: list[tuple[int, int]],
        allow_move: bool = True,
    ) -> dict:
        """Move adversary via internal tabular Q-learning policy and return transition."""
        if self.adversary_pos is None:
            return {}

        target_before = min(
            agent_positions,
            key=lambda pos: abs(pos[0] - self.adversary_pos[0]) + abs(pos[1] - self.adversary_pos[1]),
        )
        state_idx = self._encode_adversary_state(self.adversary_pos, target_before)
        action = self._select_adversary_action(state_idx)
        if not allow_move:
            action = self.ADVERSARY_ACTION_STAY

        old_dist = abs(self.adversary_pos[0] - target_before[0]) + abs(self.adversary_pos[1] - target_before[1])
        next_pos = self._apply_adversary_action(self.adversary_pos, action)
        if next_pos in self._all_obstacles():
            next_pos = self.adversary_pos
        self.adversary_pos = next_pos

        target_after = min(
            agent_positions,
            key=lambda pos: abs(pos[0] - self.adversary_pos[0]) + abs(pos[1] - self.adversary_pos[1]),
        )
        new_dist = abs(self.adversary_pos[0] - target_after[0]) + abs(self.adversary_pos[1] - target_after[1])
        caught = any(pos == self.adversary_pos for pos in agent_positions)

        next_state_idx = self._encode_adversary_state(self.adversary_pos, target_after)
        return {
            "state_idx": state_idx,
            "action": action,
            "next_state_idx": next_state_idx,
            "old_dist": old_dist,
            "new_dist": new_dist,
            "caught": caught,
        }

    def _update_adversary_q(self, transition: dict, agent_reward: float, done: bool) -> None:
        """Update adversary Q-table from saved transition and objective."""
        if not transition:
            return
        if self.adversary_freeze_episode > 0 and self._episode_count > self.adversary_freeze_episode:
            return

        if self.adversary_learning_objective == self.ADVERSARY_OBJECTIVE_ZERO_SUM:
            # Strict zero-sum: equal and opposite reward to the learning agent.
            reward = -agent_reward
        else:
            reward = self.adversary_learning_progress_reward_scale * (
                transition["old_dist"] - transition["new_dist"]
            )
            if transition["caught"]:
                reward += self.adversary_learning_catch_reward

        state_idx = int(transition["state_idx"])
        action = int(transition["action"])
        next_state_idx = int(transition["next_state_idx"])
        best_next = 0.0 if done else float(np.max(self._adversary_q_table[next_state_idx]))
        td_target = reward + self.adversary_learning_gamma * best_next
        td_error = td_target - self._adversary_q_table[state_idx, action]
        self._adversary_q_table[state_idx, action] += self.adversary_learning_alpha * td_error

    def _encode_adversary_state(self, adversary_pos: tuple[int, int], target_pos: tuple[int, int]) -> int:
        """Encode adversary/target positions into one Q-table index."""
        pos_base = self.size * self.size
        adv_idx = adversary_pos[0] * self.size + adversary_pos[1]
        tgt_idx = target_pos[0] * self.size + target_pos[1]
        return adv_idx + tgt_idx * pos_base

    def _select_adversary_action(self, state_idx: int) -> int:
        if random.random() < self._adversary_learning_epsilon:
            return random.randrange(self.ADVERSARY_ACTION_COUNT)
        values = self._adversary_q_table[state_idx]
        best_value = float(np.max(values))
        best_actions = [idx for idx, value in enumerate(values) if float(value) == best_value]
        return int(random.choice(best_actions))

    def _apply_adversary_action(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        if action == self.ADVERSARY_ACTION_STAY:
            return pos
        return self._move(pos, action)

    def _spawn_forklifts(self) -> list[tuple[int, int]]:
        if self.fixed_forklift_starts is not None:
            return list(self.fixed_forklift_starts[: self.forklift_count])
        forklifts: list[tuple[int, int]] = []
        occupied = self._reserved_cells()
        for _ in range(self.forklift_count):
            while True:
                cell = (random.randrange(self.size), random.randrange(self.size))
                if cell not in occupied:
                    forklifts.append(cell)
                    occupied.add(cell)
                    break
        return forklifts

    def _move_forklifts(self) -> None:
        if not self.forklift_positions:
            return

        new_positions: list[tuple[int, int]] = []
        occupied_next = set(self._agent_positions)
        for current in self.forklift_positions:
            adjacent_candidates: list[tuple[int, int]] = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                candidate = (nr, nc)
                if 0 <= nr < self.size and 0 <= nc < self.size and candidate not in self._all_obstacles():
                    adjacent_candidates.append(candidate)

            chosen = current
            should_move = random.random() < self.forklift_move_prob
            if should_move:
                valid_adjacent = [cell for cell in adjacent_candidates if cell not in occupied_next]
                if valid_adjacent:
                    chosen = random.choice(valid_adjacent)
            new_positions.append(chosen)
            occupied_next.add(chosen)

        self.forklift_positions = new_positions

    def _handle_pickups(self) -> float:
        bonus = 0.0
        for agent_idx, pos in enumerate(self._agent_positions):
            if self._agent_carrying[agent_idx] is not None:
                continue
            for pkg_idx, pkg_pos in enumerate(self.package_locations):
                if self._package_state[pkg_idx] == self.PACKAGE_AT_PICKUP and pos == pkg_pos:
                    self._package_state[pkg_idx] = (
                        self.PACKAGE_WITH_AGENT0 if agent_idx == 0 else self.PACKAGE_WITH_AGENT1
                    )
                    self._agent_carrying[agent_idx] = pkg_idx
                    bonus += self.pickup_reward
                    break
        return bonus

    def _handle_deliveries(self) -> float:
        bonus = 0.0
        for agent_idx, pos in enumerate(self._agent_positions):
            pkg_idx = self._agent_carrying[agent_idx]
            if pkg_idx is None:
                continue
            if pos == self.destinations[pkg_idx]:
                self._package_state[pkg_idx] = self.PACKAGE_DELIVERED
                self._agent_carrying[agent_idx] = None
                bonus += self.delivery_reward
        return bonus

    def _total_objective_distance(self) -> float:
        """Distance heuristic: to package+destination, or to destination if carrying."""
        total = 0.0
        for agent_idx, pos in enumerate(self._agent_positions):
            total += self._objective_distance_for_agent(agent_idx, pos)
        return total

    def _objective_distance_for_agent(self, agent_idx: int, pos: tuple[int, int]) -> float:
        carried_pkg = self._agent_carrying[agent_idx]
        if carried_pkg is not None:
            dst = self.destinations[carried_pkg]
            return abs(pos[0] - dst[0]) + abs(pos[1] - dst[1])

        candidates: list[float] = []
        for pkg_idx, pkg in enumerate(self.package_locations):
            if self._package_state[pkg_idx] != self.PACKAGE_AT_PICKUP:
                continue
            dst = self.destinations[pkg_idx]
            to_pkg = abs(pos[0] - pkg[0]) + abs(pos[1] - pkg[1])
            pkg_to_dst = abs(pkg[0] - dst[0]) + abs(pkg[1] - dst[1])
            candidates.append(float(to_pkg + pkg_to_dst))
        return min(candidates) if candidates else 0.0

    def _generate_packages_and_destinations(self) -> None:
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
            if cells.isdisjoint(self.starts) and cells.isdisjoint(self.shelf_obstacles):
                return cells

    def _sample_spills(self) -> set[tuple[int, int]]:
        spills: set[tuple[int, int]] = set()
        occupied = self._reserved_cells()
        while len(spills) < self.spill_count:
            cell = (random.randrange(self.size), random.randrange(self.size))
            if cell not in occupied:
                spills.add(cell)
                occupied.add(cell)
        return spills

    def _decode_joint_action(self, action: int) -> list[int]:
        actions: list[int] = []
        remaining = action
        for _ in range(self.agent_count):
            actions.append(remaining % self.n_actions)
            remaining //= self.n_actions
        return actions

    def _all_obstacles(self) -> set[tuple[int, int]]:
        return self.shelf_obstacles | self.spill_obstacles

    def _dynamic_slot_count(self) -> int:
        if self.adversary_enabled:
            return 1
        return max(1, self.forklift_count)

    def _dynamic_positions_for_encoding(self) -> list[tuple[int, int]]:
        if self.adversary_enabled:
            if not self._adversary_positions:
                return [(0, 0)]
            if len(self._adversary_positions) == 1:
                return [self._adversary_positions[0]]
            # Multiple adversaries: encode nearest to agent 0 to keep state space unchanged.
            agent_pos = self._agent_positions[0] if self._agent_positions else (0, 0)
            nearest = min(
                self._adversary_positions,
                key=lambda p: abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]),
            )
            return [nearest]
        positions = list(self.forklift_positions)
        while len(positions) < self._dynamic_slot_count():
            positions.append((0, 0))
        return positions

    def _proximity_direction_code(
        self,
        entity_pos: tuple[int, int] | None,
        agent_pos: tuple[int, int],
        radius: int,
    ) -> int:
        """Return 5-bucket directional proximity code: 0=not-nearby, 1=N, 2=S, 3=E, 4=W."""
        if entity_pos is None:
            return 0
        dr = entity_pos[0] - agent_pos[0]
        dc = entity_pos[1] - agent_pos[1]
        if abs(dr) > radius or abs(dc) > radius:
            return 0
        if abs(dr) >= abs(dc):
            return 1 if dr < 0 else 2  # N or S
        return 3 if dc > 0 else 4      # E or W

    def _dynamic_proximity_codes(self, agent_pos: tuple[int, int]) -> list[int]:
        """Return one proximity code per dynamic slot (consistent across adversary/nature)."""
        if self.adversary_enabled:
            return [self._proximity_direction_code(self.adversary_pos, agent_pos, self.adversary_proximity_radius)]
        if self.forklift_positions:
            nearest = min(
                self.forklift_positions,
                key=lambda fp: abs(fp[0] - agent_pos[0]) + abs(fp[1] - agent_pos[1]),
            )
            return [self._proximity_direction_code(nearest, agent_pos, self.adversary_proximity_radius)]
        return [0]

    def _octant_direction(self, dr: int, dc: int) -> int:
        """Encode relative direction as 0–8: 0=same, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW."""
        if dr == 0 and dc == 0:
            return 0
        if dr < 0 and dc == 0:
            return 1
        if dr < 0 and dc > 0:
            return 2
        if dr == 0 and dc > 0:
            return 3
        if dr > 0 and dc > 0:
            return 4
        if dr > 0 and dc == 0:
            return 5
        if dr > 0 and dc < 0:
            return 6
        if dr == 0 and dc < 0:
            return 7
        return 8  # dr < 0 and dc < 0 → NW

    def _encode_coarse_destination_directions(self, state: tuple[tuple[int, int], ...]) -> int:
        """Encode octant direction from agent to each package and its destination (9^(2*num_packages) values)."""
        ref_row, ref_col = state[0]
        code = 0
        multiplier = 1
        for pkg, dst in zip(self.package_locations, self.destinations):
            pkg_dir = self._octant_direction(pkg[0] - ref_row, pkg[1] - ref_col)
            dst_dir = self._octant_direction(dst[0] - ref_row, dst[1] - ref_col)
            code += pkg_dir * multiplier
            multiplier *= 9
            code += dst_dir * multiplier
            multiplier *= 9
        return code

    def _reserved_cells(self) -> set[tuple[int, int]]:
        return set(self.starts) | set(self.shelf_obstacles) | set(self.package_locations) | set(self.destinations)

    def render(self, path: str | None = None) -> None:
        """Render a static view of the environment."""
        grid = np.zeros((self.size, self.size), dtype=int)
        for row, col in self.shelf_obstacles:
            grid[row, col] = self.RENDER_SHELF
        for row, col in self.spill_obstacles:
            grid[row, col] = self.RENDER_SPILL

        grid[self.starts[0]] = self.RENDER_START0
        if self.agent_count > 1:
            grid[self.starts[1]] = self.RENDER_START1

        for pkg in self.package_locations:
            grid[pkg] = self.RENDER_PACKAGE
        for dst in self.destinations:
            grid[dst] = self.RENDER_DESTINATION

        for adv_pos in self._adversary_positions:
            grid[adv_pos] = self.RENDER_ADVERSARY
        for forklift in self.forklift_positions:
            grid[forklift] = self.RENDER_FORKLIFT

        colors = np.array(
            [
                [1.0, 1.0, 1.0],
                [0.2, 0.6, 1.0],
                [0.2, 0.4, 0.8],
                [1.0, 0.8, 0.2],
                [0.2, 0.8, 0.2],
                [0.3, 0.3, 0.3],
                [0.7, 0.2, 0.2],
                [0.9, 0.0, 0.9],
                [1.0, 0.45, 0.0],
            ]
        )

        plt.figure(figsize=(4.8, 4.8))
        plt.imshow(colors[grid], interpolation="none")
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.grid(which="both", color="black", linewidth=0.5)

        title = "Nature: Shelves + Random Forklift"
        if self.scenario == self.SCENARIO_ADVERSARY:
            title = "Adversary: Shelves + Pursuit Adversary"
        plt.title(title)

        handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[self.RENDER_START0], markersize=10, label="Start"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[self.RENDER_PACKAGE], markersize=10, label="Package"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[self.RENDER_DESTINATION], markersize=10, label="Destination"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[self.RENDER_SHELF], markersize=10, label="Shelf"),
        ]
        if self.scenario == self.SCENARIO_NATURE:
            handles.append(
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[self.RENDER_FORKLIFT], markersize=10, label="Forklift")
            )
        if self.scenario == self.SCENARIO_ADVERSARY:
            handles.append(
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[self.RENDER_ADVERSARY], markersize=10, label="Adversary")
            )

        # Overlay learning agent markers.
        # Colors: yellow=carrying, purple=successful delivery, red=caught collision.
        delivered_all = all(state == self.PACKAGE_DELIVERED for state in self._package_state)
        caught_collision = self._terminal_status == "caught"
        agent0 = self._agent_positions[0]
        if caught_collision:
            agent0_color = "red"
        elif delivered_all:
            agent0_color = "purple"
        else:
            agent0_color = "yellow" if self._agent_carrying[0] is not None else "black"
        plt.scatter([agent0[1]], [agent0[0]], c=agent0_color, marker="o", s=65, zorder=10)
        if self.agent_count > 1:
            agent1 = self._agent_positions[1]
            if caught_collision:
                agent1_color = "red"
            elif delivered_all:
                agent1_color = "purple"
            else:
                agent1_color = "yellow" if self._agent_carrying[1] is not None else "black"
            plt.scatter([agent1[1]], [agent1[0]], c=agent1_color, marker="x", s=65, zorder=10)

        plt.legend(handles=handles, loc="upper right", fontsize="small")
        plt.tight_layout()

        if path:
            plt.savefig(path)
        plt.close()

    def delivered_package_count(self) -> int:
        """Return how many packages are delivered in the current episode state."""
        return sum(1 for state in self._package_state if state == self.PACKAGE_DELIVERED)
