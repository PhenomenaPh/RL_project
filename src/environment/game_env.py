"""Gym environment wrapper for the strategy game."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.environment.mini_strat_game import Player, StrategyGame, encode_simple_colors


@dataclass
class EpisodeMetrics:
    """Tracks metrics for a single episode."""

    start_time: float
    total_actions: int = 0
    successful_actions: int = 0
    resources_collected: float = 0.0
    enemy_units_destroyed: int = 0
    own_units_destroyed: int = 0
    initial_units: int = 0
    territory_cells: int = 0
    total_cells: int = 0


class StrategyGameEnv(gym.Env):
    """OpenAI Gym environment wrapper for the strategy game."""

    def __init__(self, opponent: Player):
        """Initialize environment."""
        super().__init__()

        # Action space: [worker, pikeman, swordsman, archer, medic, knight, x_pos, y_pos]
        self.action_space = gym.spaces.Box(
            low=np.zeros(8), high=np.ones(8), dtype=np.float32
        )

        # Observation space: encoded image (474) + statistical features (14)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(474 + 14,), dtype=np.float32
        )

        self.opponent = opponent
        self.game: Optional[StrategyGame] = None
        self.current_step = 0
        self.max_steps = 1000
        self.episode_metrics: Optional[EpisodeMetrics] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Initialize new game
        self.game = StrategyGame()
        self.current_step = 0

        # Initialize metrics
        self.episode_metrics = EpisodeMetrics(
            start_time=time.time(), total_cells=self.game.field.size
        )

        # Count initial units
        for unit in self.game.units:
            if unit.team == 1:  # Our team
                self.episode_metrics.initial_units += 1

        # Get initial observation
        obs = self._get_observation()

        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        assert self.game is not None
        assert self.episode_metrics is not None

        # Store state before action
        prev_money = self.game.money[1]
        prev_units = self._count_units()

        # Take agent action (team 1)
        self.game.action(1, action.tolist())
        self.episode_metrics.total_actions += 1

        # Take opponent action (team 0)
        img = self.game.step_env()
        opponent_action = self.opponent.act(img, self.game, team=0)
        self.game.action(0, opponent_action)

        # Update game state
        img = self.game.step_env()

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Update metrics
        self._update_metrics(prev_money, prev_units)

        # Check if episode is done
        self.current_step += 1
        done = self._check_done()
        truncated = self.current_step >= self.max_steps

        # Get info
        info = self._get_info()

        return obs, reward, done, truncated, info

    def _count_units(self) -> Dict[str, Dict[int, int]]:
        """Count units for each team."""
        assert self.game is not None

        counts = {"total": {0: 0, 1: 0}, "alive": {0: 0, 1: 0}}

        for unit in self.game.units:
            counts["total"][unit.team] += 1
            if unit.hp > 0:
                counts["alive"][unit.team] += 1

        return counts

    def _update_metrics(self, prev_money: float, prev_units: Dict[str, Dict[int, int]]):
        """Update episode metrics after an action."""
        assert self.game is not None
        assert self.episode_metrics is not None

        # Check if action was successful
        current_units = self._count_units()
        money_gained = self.game.money[1] - prev_money
        units_gained = current_units["alive"][1] - prev_units["alive"][1]
        enemy_units_destroyed = prev_units["alive"][0] - current_units["alive"][0]

        if money_gained > 0 or units_gained > 0 or enemy_units_destroyed > 0:
            self.episode_metrics.successful_actions += 1

        # Update resource collection
        self.episode_metrics.resources_collected += money_gained

        # Update unit destruction counts
        self.episode_metrics.enemy_units_destroyed += enemy_units_destroyed
        own_units_destroyed = prev_units["alive"][1] - current_units["alive"][1]
        self.episode_metrics.own_units_destroyed += own_units_destroyed

        # Update territory control
        territory = 0
        for unit in self.game.units:
            if unit.hp > 0 and unit.team == 1:
                # Count cells in unit's control radius
                x, y = int(unit.x), int(unit.y)
                radius = int(unit.att_range)
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx * dx + dy * dy <= radius * radius:
                            territory += 1
        self.episode_metrics.territory_cells = territory

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict for current state."""
        assert self.game is not None
        assert self.episode_metrics is not None

        # Count units
        our_units = 0
        enemy_units = 0
        our_workers = 0
        enemy_workers = 0

        for unit in self.game.units:
            if unit.hp <= 0:
                continue
            if unit.team == 1:
                our_units += 1
                if unit.type == "worker":
                    our_workers += 1
            else:
                enemy_units += 1
                if unit.type == "worker":
                    enemy_workers += 1

        # Calculate metrics
        episode_time = time.time() - self.episode_metrics.start_time
        success_rate = self.episode_metrics.successful_actions / max(
            1, self.episode_metrics.total_actions
        )
        unit_survival_rate = our_units / max(1, self.episode_metrics.initial_units)
        destruction_ratio = self.episode_metrics.enemy_units_destroyed / max(
            1, self.episode_metrics.own_units_destroyed
        )
        territory_control = (
            self.episode_metrics.territory_cells / self.episode_metrics.total_cells
        )
        total_resources = (
            self.game.money[0] + self.game.money[1] + self.game.field.sum()
        )
        resource_share = self.episode_metrics.resources_collected / max(
            1, total_resources
        )

        return {
            "our_money": self.game.money[1],
            "enemy_money": self.game.money[0],
            "our_units": our_units,
            "enemy_units": enemy_units,
            "our_workers": our_workers,
            "enemy_workers": enemy_workers,
            "remaining_resources": self.game.field.sum(),
            # Additional metrics
            "episode_time": episode_time,
            "success_rate": success_rate,
            "unit_survival_rate": unit_survival_rate,
            "destruction_ratio": destruction_ratio,
            "territory_control": territory_control,
            "resource_share": resource_share,
        }

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get game image
        img = self.game.step_env()

        # Encode image
        encoded_img = encode_simple_colors(np.array(img))

        # Get statistical features
        stats = np.zeros(14)
        stats[12:14] = self.game.money  # Team money

        # Count units
        for unit in self.game.units:
            if unit.hp <= 0:
                continue

            idx = 0 if unit.team == 0 else 6
            if unit.type == "worker":
                stats[idx] += 1
            elif unit.type == "pikeman":
                stats[idx + 1] += 1
            elif unit.type == "swordsman":
                stats[idx + 2] += 1
            elif unit.type == "archer":
                stats[idx + 3] += 1
            elif unit.type != "medic":
                stats[idx + 4] += 1
            elif unit.type == "knight":
                stats[idx + 5] += 1

        # Combine features
        obs = np.concatenate([encoded_img, stats])
        return obs.astype(np.float32)

    def _calculate_reward(self) -> float:
        """Calculate reward for current step."""
        reward = 0.0

        # Money reward
        money_diff = self.game.money[1] - self.game.money[0]
        reward += 0.1 * money_diff

        # Unit advantage reward
        our_units = 0
        enemy_units = 0
        for unit in self.game.units:
            if unit.hp <= 0:
                continue
            if unit.team == 1:
                our_units += 1
            else:
                enemy_units += 1

        unit_diff = our_units - enemy_units
        reward += 0.5 * unit_diff

        # Victory reward
        if enemy_units == 0 and our_units > 0:
            reward += 10.0
        elif our_units == 0 and enemy_units > 0:
            reward -= 10.0

        return reward

    def _check_done(self) -> bool:
        """Check if episode is done."""
        # Count units
        our_units = 0
        enemy_units = 0
        for unit in self.game.units:
            if unit.hp <= 0:
                continue
            if unit.team == 1:
                our_units += 1
            else:
                enemy_units += 1

        # Game is done if either team is eliminated
        return our_units == 0 or enemy_units == 0

    def render(self) -> np.ndarray:
        """Render current game state."""
        return np.array(self.game.step_env())
