"""Gym environment wrapper for the strategy game."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.environment.mini_strat_game import (
    Player,
    StrategyGame,
    encode_simple_colors,
    FIELD_SIZE,
)


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
        """Initialize environment.
        
        Args:
            opponent: Player object that will control the opponent team
        """
        super().__init__()
        
        # Game state
        self.game: Optional[StrategyGame] = None
        self.opponent = opponent
        self.episode_metrics: Optional[EpisodeMetrics] = None
        self._prev_money = 0  # Track previous money for reward calculation
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 1000  # Maximum steps per episode
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(8,), dtype=np.float32
        )  # [unit_probs(6), x, y]
        
        # Observation space: encoded image + stats (14)
        # Image is encoded to 474 values (see encode_simple_colors function)
        encoded_img_size = 474  # This is the size after encoding
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(encoded_img_size + 14,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation and info dict
        """
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize new game
        self.game = StrategyGame()
        self._prev_money = self.game.money[1]  # Initialize previous money tracker
        
        # Reset step counter
        self.current_step = 0
        
        # Reset metrics
        self.episode_metrics = EpisodeMetrics(start_time=time.time())
        self.episode_metrics.initial_units = 0
        self.episode_metrics.total_cells = FIELD_SIZE * FIELD_SIZE
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        assert self.game is not None
        assert self.episode_metrics is not None

        # Validate and format action
        action = np.array(action).flatten()  # Ensure action is 1D
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        if len(action) != 8:
            print(f"Warning: Invalid action shape {action.shape}, expected (8,)")
            # Pad action with zeros if too short, or truncate if too long
            padded_action = np.zeros(8, dtype=np.float32)
            padded_action[:min(len(action), 8)] = action[:8]
            action = padded_action

        # Store state before action
        prev_money = self.game.money[1]
        prev_units = self._count_units(team=1)  # Count our units before action

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

        # Update step counter
        self.current_step += 1

        # Check if episode is done
        done = self._check_done()
        truncated = self.current_step >= self.max_steps

        # Get info
        info = self._get_info()

        return obs, reward, done, truncated, info

    def _count_units(self, team: Optional[int] = None) -> int:
        """Count units for a specific team or all units.
        
        Args:
            team: Team number (0 or 1) or None to count all units
            
        Returns:
            Number of units alive for the specified team or total units if team is None
        """
        assert self.game is not None
        
        count = 0
        for unit in self.game.units:
            if unit.hp <= 0:  # Skip dead units
                continue
            if team is None or unit.team == team:
                count += 1
                
        return count

    def _count_territory(self) -> int:
        """Count territory controlled by our units.
        
        Returns:
            Number of cells under our control
        """
        assert self.game is not None
        
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
        return territory

    def _update_metrics(self, prev_money: float, prev_units: int):
        """Update episode metrics after an action."""
        assert self.game is not None
        assert self.episode_metrics is not None

        # Check if action was successful
        current_units = self._count_units(team=1)  # Our units
        enemy_units = self._count_units(team=0)  # Enemy units
        money_gained = self.game.money[1] - prev_money
        units_gained = current_units - prev_units
        enemy_units_destroyed = prev_units - enemy_units

        if money_gained > 0 or units_gained > 0 or enemy_units_destroyed > 0:
            self.episode_metrics.successful_actions += 1

        # Update resource collection
        self.episode_metrics.resources_collected += money_gained

        # Update unit destruction counts
        self.episode_metrics.enemy_units_destroyed += enemy_units_destroyed
        own_units_destroyed = prev_units - current_units
        self.episode_metrics.own_units_destroyed += own_units_destroyed

        # Update territory control
        territory = self._count_territory()
        self.episode_metrics.territory_cells = territory

    def _get_info(self) -> Dict[str, Any]:
        """Get current game info/metrics."""
        assert self.game is not None
        assert self.episode_metrics is not None

        # Calculate success rate
        success_rate = (
            self.episode_metrics.successful_actions / self.episode_metrics.total_actions
            if self.episode_metrics.total_actions > 0
            else 0.0
        )

        # Calculate unit counts
        our_units = self._count_units(team=1)
        enemy_units = self._count_units(team=0)
        total_units = our_units + enemy_units

        # Calculate survival rate
        survival_rate = (
            our_units / self.episode_metrics.initial_units
            if self.episode_metrics.initial_units > 0
            else 1.0
        )

        # Calculate territory control
        territory = self._count_territory()
        territory_control = (
            territory / self.episode_metrics.total_cells
            if self.episode_metrics.total_cells > 0
            else 0.0
        )

        # Calculate gold metrics
        our_gold = float(self.game.money[1])
        enemy_gold = float(self.game.money[0])
        total_gold = our_gold + enemy_gold + 1e-8  # Avoid division by zero
        
        # Calculate gold ratios
        gold_ratio = our_gold / total_gold if total_gold > 0 else 0.0
        gold_vs_opponent = our_gold / (enemy_gold + 1e-8)  # Avoid division by zero

        return {
            "success_rate": success_rate,
            "episode_time": time.time() - self.episode_metrics.start_time,
            "unit_survival_rate": survival_rate,
            "territory_control": territory_control,
            "our_units": our_units,
            "enemy_units": enemy_units,
            "our_gold": our_gold,
            "enemy_gold": enemy_gold,
            "gold_ratio": gold_ratio,
            "gold_vs_opponent": gold_vs_opponent,
            "total_gold": total_gold
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
        assert self.game is not None
        reward = 0.0

        # Count units
        our_units = 0
        enemy_units = 0
        our_workers = 0
        for unit in self.game.units:
            if unit.hp <= 0:
                continue
            if unit.team == 1:
                our_units += 1
                if unit.type == "worker":
                    our_workers += 1
            else:
                enemy_units += 1

        # Gold rewards (primary objective)
        gold_ratio = self.game.money[1] / max(1, self.game.money[0])
        reward += 1.0 * (gold_ratio - 1.0)  # Doubled reward for having more gold
        
        # Direct reward for gold collection
        if self.game.money[1] > self._prev_money:
            reward += 0.5  # Immediate reward for collecting gold
        self._prev_money = self.game.money[1]
        
        # Worker reward
        if our_workers == 0 and self.game.money[1] < 100:
            reward -= 0.5  # Penalty for having no workers when gold is low

        # Unit advantage reward (secondary objective)
        unit_ratio = our_units / max(1, enemy_units)
        reward += 0.3 * (unit_ratio - 1.0)  # Reward for having more units than opponent

        # Victory rewards
        if our_units == 0 and self.game.money[1] < self.game.money[0]:
            # Lost both economically and militarily
            reward -= 10.0
        elif enemy_units == 0 and self.game.money[0] < self.game.money[1]:
            # Won both economically and militarily
            reward += 10.0
        elif self.game.money[1] > 2 * self.game.money[0] and our_units > enemy_units:
            # Economic and military dominance
            reward += 5.0
        elif self.game.money[0] > 2 * self.game.money[1] and enemy_units > our_units:
            # Enemy economic and military dominance
            reward -= 5.0

        return reward

    def _check_done(self) -> bool:
        """Check if episode is done."""
        assert self.game is not None

        # Count units and gold
        our_units = 0
        enemy_units = 0
        for unit in self.game.units:
            if unit.hp <= 0:
                continue
            if unit.team == 1:
                our_units += 1
            else:
                enemy_units += 1

        our_gold = self.game.money[1]
        enemy_gold = self.game.money[0]

        # Game is done if:
        # 1. One team has no units AND less gold
        # 2. One team has significantly more gold (2x) and more units
        # 3. Maximum steps reached (handled in step method)
        
        if our_units == 0 and our_gold < enemy_gold:
            return True
        if enemy_units == 0 and enemy_gold < our_gold:
            return True
        
        # Check for economic and military dominance
        if our_gold > 2 * enemy_gold and our_units > enemy_units:
            return True
        if enemy_gold > 2 * our_gold and enemy_units > our_units:
            return True

        return False

    def render(self) -> np.ndarray:
        """Render current game state."""
        return np.array(self.game.step_env())
