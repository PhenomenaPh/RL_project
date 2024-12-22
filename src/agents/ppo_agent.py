"""PPO agent implementation for the strategy game."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None


class ActorCriticPolicy(nn.Module):
    """Combined actor-critic network."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )

        # Policy head (actor)
        self.mean = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Value head (critic)
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(x)
        mean = self.mean(features)
        std = torch.exp(self.log_std)
        value = self.value(features)
        return mean, std, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, values = self(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values.squeeze(-1), entropy

    def predict(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mean, std, _ = self(obs)
            dist = Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs


class PPOAgent:
    """PPO agent for training and inference."""

    def __init__(
        self, state_dim: int, action_dim: int, config: PPOConfig = PPOConfig()
    ):
        """Initialize PPO agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: PPO configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy
        self.policy = ActorCriticPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)

        # Initialize buffers
        self.reset_buffers()

    def reset_buffers(self):
        """Reset experience buffers."""
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []
        self.done_buffer = []

    def collect_rollouts(self, env: gym.Env) -> Dict[str, float]:
        """Collect experience using current policy."""
        self.reset_buffers()
        obs, _ = env.reset()

        episode_rewards = []
        current_rewards = []

        for step in range(self.config.n_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Get action from policy
            with torch.no_grad():
                action, log_prob = self.policy.predict(obs_tensor)
                _, _, value = self.policy(obs_tensor)

            # Execute action
            next_obs, reward, done, truncated, info = env.step(action.cpu().numpy()[0])

            # Store experience
            self.obs_buffer.append(obs)
            self.action_buffer.append(action[0])  # Remove batch dimension
            self.reward_buffer.append(reward)
            self.value_buffer.append(value[0])  # Remove batch dimension
            self.log_prob_buffer.append(log_prob[0])  # Remove batch dimension
            self.done_buffer.append(done)

            # Track rewards
            current_rewards.append(reward)

            if done or truncated:
                episode_rewards.append(float(np.sum(current_rewards)))  # Use numpy sum
                current_rewards = []
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Convert buffers to tensors more efficiently
        self.obs_buffer = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        self.action_buffer = torch.stack(self.action_buffer)
        self.reward_buffer = torch.FloatTensor(np.array(self.reward_buffer)).to(self.device)
        self.value_buffer = torch.cat(self.value_buffer)
        self.log_prob_buffer = torch.stack(self.log_prob_buffer)
        self.done_buffer = torch.FloatTensor(np.array(self.done_buffer)).to(self.device)

        # Compute advantages and returns
        advantages = self._compute_advantages()
        returns = advantages + self.value_buffer

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "mean_reward": float(np.mean(episode_rewards) if episode_rewards else 0.0),
            "min_reward": float(np.min(episode_rewards) if episode_rewards else 0.0),
            "max_reward": float(np.max(episode_rewards) if episode_rewards else 0.0),
            "success_rate": float(info.get("success_rate", 0.0)),
            "episode_time": float(info.get("episode_time", 0.0)),
            "unit_survival_rate": float(info.get("unit_survival_rate", 0.0)),
            "destruction_ratio": float(info.get("destruction_ratio", 0.0)),
            "territory_control": float(info.get("territory_control", 0.0)),
            "resource_share": float(info.get("resource_share", 0.0)),
            "our_units": float(info.get("our_units", 0)),
            "enemy_units": float(info.get("enemy_units", 0)),
        }

    def _compute_advantages(self) -> torch.Tensor:
        """Compute GAE advantages."""
        # Get last value for bootstrapping
        with torch.no_grad():
            last_obs = (
                torch.FloatTensor(self.obs_buffer[-1]).unsqueeze(0).to(self.device)
            )
            _, _, last_value = self.policy(last_obs)
            last_value = last_value.squeeze()

        advantages = torch.zeros_like(self.reward_buffer)
        last_gae = 0

        # Compute GAE
        for t in reversed(range(len(self.reward_buffer))):
            if t == len(self.reward_buffer) - 1:
                next_non_terminal = 1.0 - self.done_buffer[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.done_buffer[t]
                next_value = self.value_buffer[t + 1]

            delta = (
                self.reward_buffer[t]
                + self.config.gamma * next_value * next_non_terminal
                - self.value_buffer[t]
            )

            last_gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * next_non_terminal
                * last_gae
            )
            advantages[t] = last_gae

        return advantages

    def train_epoch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Train one epoch on collected experience."""
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            obs, actions, old_log_probs, advantages, returns
        )
        
        # Calculate number of minibatches
        n_samples = len(dataset)
        batch_size = min(self.config.batch_size, n_samples)
        n_batches = n_samples // batch_size
        if n_batches == 0:
            n_batches = 1
            batch_size = n_samples
        
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        # Track metrics
        metrics: Dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "kl_div": [],
            "clip_fraction": [],
        }

        # Train for n_epochs
        for _ in range(self.config.n_epochs):
            for batch in loader:
                (
                    batch_obs,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                ) = batch

                # Evaluate actions under current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Calculate policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                policy_loss1 = -batch_advantages * ratio
                policy_loss2 = -batch_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                )
                policy_loss = torch.mean(torch.max(policy_loss1, policy_loss2))

                # Calculate value loss
                value_loss = torch.mean(torch.square(values - batch_returns))

                # Calculate entropy loss
                entropy_loss = -torch.mean(entropy)

                # Calculate total loss
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    metrics["policy_loss"].append(policy_loss.item())
                    metrics["value_loss"].append(value_loss.item())
                    metrics["entropy_loss"].append(entropy_loss.item())
                    metrics["kl_div"].append(
                        torch.mean(batch_old_log_probs - log_probs).item()
                    )
                    metrics["clip_fraction"].append(
                        torch.mean(
                            (torch.abs(ratio - 1) > self.config.clip_range).float()
                        ).item()
                    )

                # Early stopping based on KL divergence
                if (
                    self.config.target_kl is not None
                    and metrics["kl_div"][-1] > 1.5 * self.config.target_kl
                ):
                    break

        # Average metrics
        return {k: np.mean(v) for k, v in metrics.items()}

    def save(self, path: str) -> None:
        """Save policy state dict."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """Load policy state dict."""
        self.policy.load_state_dict(torch.load(path))
