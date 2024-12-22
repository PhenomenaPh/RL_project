"""Training script for PPO agent."""

import argparse
import os
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.agents.ppo_agent import PPOAgent, PPOConfig
from src.environment.game_env import StrategyGameEnv
from src.environment.mini_strat_game import (
    AdaptivePlayer,
    ArcherPlayer,
    ArmyPlayer,
    BalancedPlayer,
    WorkerRushPlayer,
)


class MetricsTracker:
    """Tracks and computes training metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.episode_rewards = deque(maxlen=self.window_size)
        self.episode_lengths = deque(maxlen=self.window_size)
        self.win_rates = deque(maxlen=self.window_size)
        self.success_rates = deque(maxlen=self.window_size)
        self.episode_times = deque(maxlen=self.window_size)
        self.survival_rates = deque(maxlen=self.window_size)
        self.destruction_ratios = deque(maxlen=self.window_size)
        self.territory_controls = deque(maxlen=self.window_size)
        self.resource_shares = deque(maxlen=self.window_size)
        self.q_losses = deque(maxlen=self.window_size)

        self.baseline_win_rate = None
        self.baseline_reward = None
        self.episode_count = 0

    def update(
        self,
        episode_reward: float,
        episode_length: int,
        info: Dict,
        q_loss: Optional[float] = None,
    ):
        """Update metrics with new episode results."""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        # Game metrics
        win = info["our_units"] > 0 and info["enemy_units"] == 0
        self.win_rates.append(float(win))
        self.success_rates.append(info["success_rate"])
        self.episode_times.append(info["episode_time"])
        self.survival_rates.append(info["unit_survival_rate"])
        self.destruction_ratios.append(info["destruction_ratio"])
        self.territory_controls.append(info["territory_control"])
        self.resource_shares.append(info["resource_share"])

        # Algorithm metrics
        if q_loss is not None:
            self.q_losses.append(q_loss)

        self.episode_count += 1

        # Set baseline after first 100 episodes
        if self.episode_count == 100:
            self.baseline_win_rate = self.get_win_rate()
            self.baseline_reward = self.get_mean_reward()

    def get_metrics(self) -> Dict:
        """Get current metrics."""
        metrics = {
            "reward/mean": self.get_mean_reward(),
            "reward/std": self.get_std_reward(),
            "length/mean": np.mean(self.episode_lengths),
            "win_rate": self.get_win_rate(),
            "success_rate/mean": np.mean(self.success_rates),
            "episode_time/mean": np.mean(self.episode_times),
            "survival_rate/mean": np.mean(self.survival_rates),
            "destruction_ratio/mean": np.mean(self.destruction_ratios),
            "territory_control/mean": np.mean(self.territory_controls),
            "resource_share/mean": np.mean(self.resource_shares),
        }

        # Add Q-loss if available
        if len(self.q_losses) > 0:
            metrics["loss/q_value"] = np.mean(self.q_losses)

        # Add improvement metrics after baseline is set
        if self.baseline_win_rate is not None:
            metrics["improvement/win_rate"] = (
                self.get_win_rate() - self.baseline_win_rate
            )
            metrics["improvement/reward"] = (
                self.get_mean_reward() - self.baseline_reward
            )

        return metrics

    def get_win_rate(self) -> float:
        """Get current win rate."""
        return np.mean(self.win_rates) if len(self.win_rates) > 0 else 0.0

    def get_mean_reward(self) -> float:
        """Get mean episode reward."""
        return np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0.0

    def get_std_reward(self) -> float:
        """Get standard deviation of episode rewards."""
        return np.std(self.episode_rewards) if len(self.episode_rewards) > 0 else 0.0

    def plot_metrics(self, save_dir: str = "plots") -> None:
        """Generate and save plots of training metrics.

        Args:
            save_dir: Directory to save the plots in.
        """
        import os

        import matplotlib.pyplot as plt

        # Create plots directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Common plot settings
        plt.style.use("seaborn")
        episodes = range(1, len(self.episode_rewards) + 1)

        # Plot 1: Training Progress (Rewards and Win Rate)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Rewards
        ax1.plot(episodes, self.episode_rewards, label="Episode Reward")
        ax1.set_title("Training Progress - Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.legend()

        # Win Rate
        ax2.plot(episodes, [float(x) for x in self.win_rates], label="Win Rate")
        ax2.set_title("Training Progress - Win Rate")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Win Rate")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_progress.png"))
        plt.close()

        # Plot 2: Game Performance Metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        ax1.plot(episodes, self.success_rates, label="Success Rate")
        ax1.set_title("Success Action Rate")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Rate")
        ax1.legend()

        ax2.plot(episodes, self.survival_rates, label="Survival Rate")
        ax2.set_title("Unit Survival Rate")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Rate")
        ax2.legend()

        ax3.plot(episodes, self.destruction_ratios, label="Destruction Ratio")
        ax3.set_title("Enemy/Own Units Destroyed Ratio")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Ratio")
        ax3.legend()

        ax4.plot(episodes, self.territory_controls, label="Territory Control")
        ax4.set_title("Territory Control")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Control %")
        ax4.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "game_metrics.png"))
        plt.close()

        # Plot 3: Resource and Time Metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(episodes, self.resource_shares, label="Resource Share")
        ax1.set_title("Resource Collection Rate")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Share")
        ax1.legend()

        ax2.plot(episodes, self.episode_times, label="Episode Time")
        ax2.set_title("Average Episode Time")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Time (s)")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "resource_time_metrics.png"))
        plt.close()

        # Plot 4: Algorithm Metrics (Q-Loss)
        if len(self.q_losses) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(self.q_losses) + 1), self.q_losses, label="Q-Loss")
            plt.title("Training Loss")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "q_loss.png"))
            plt.close()


def train(
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = 0.015,
    seed: Optional[int] = None,
    device: str = "auto",
    log_interval: int = 1,
    save_path: str = "models",
    opponent: str = "balanced",
) -> PPOAgent:
    """Train PPO agent on strategy game."""
    try:
        print("Starting training...")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {device}")
        print(f"Opponent: {opponent}")

        # Set random seed
        if seed is not None:
            print(f"Setting random seed: {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Create save directory
        print(f"Creating save directory: {save_path}")
        os.makedirs(save_path, exist_ok=True)

        # Initialize opponent
        print("Initializing opponent...")
        opponents = {
            "balanced": BalancedPlayer(),
            "adaptive": AdaptivePlayer(),
            "archer": ArcherPlayer(),
            "worker_rush": WorkerRushPlayer(),
            "army": ArmyPlayer(),
        }
        if opponent not in opponents:
            raise ValueError(f"Unknown opponent type: {opponent}")
        opponent_player = opponents[opponent]

        # Create environment
        print("Creating environment...")
        env = StrategyGameEnv(opponent_player)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Get state and action dimensions
        state_dim = env.observation_space.shape[0] if env.observation_space.shape else 0
        action_dim = env.action_space.shape[0] if env.action_space.shape else 0

        # Initialize agent
        print("Initializing agent...")
        config = PPOConfig(
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
        )

        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
        )
        print(f"Agent initialized with state_dim={state_dim}, action_dim={action_dim}")

        # Initialize metrics tracker
        print("Initializing metrics tracker...")
        metrics = MetricsTracker()

        # Training loop
        print("Starting training loop...")
        timesteps_elapsed = 0
        try:
            # Create progress bar for total training
            pbar = tqdm(
                total=total_timesteps,
                desc="Training Progress",
                unit="steps",
                position=0,
                leave=True,
            )

            while timesteps_elapsed < total_timesteps:
                # Collect rollout
                rollout_metrics = agent.collect_rollouts(env)
                timesteps_elapsed += agent.config.n_steps
                pbar.update(agent.config.n_steps)

                # Train on collected experience
                train_metrics = agent.train_epoch(
                    agent.obs_buffer,
                    agent.action_buffer,
                    agent.log_prob_buffer,
                    agent._compute_advantages(),
                    agent._compute_advantages() + agent.value_buffer,
                )

                # Update metrics
                metrics.update(
                    episode_reward=rollout_metrics["mean_reward"],
                    episode_length=agent.config.n_steps,
                    info=rollout_metrics,
                    q_loss=train_metrics.get("value_loss"),
                )

                # Log metrics
                if timesteps_elapsed % (log_interval * agent.config.n_steps) == 0:
                    current_metrics = metrics.get_metrics()
                    pbar.set_postfix(
                        {
                            "reward": f"{current_metrics['reward/mean']:.2f}±{current_metrics['reward/std']:.2f}",
                            "win_rate": f"{current_metrics['win_rate']:.1%}",
                            "success_rate": f"{current_metrics['success_rate/mean']:.1%}",
                        }
                    )

                # Save checkpoint every 100k steps
                if timesteps_elapsed % 100000 == 0:
                    checkpoint_path = os.path.join(
                        save_path, f"ppo_checkpoint_{timesteps_elapsed}.pt"
                    )
                    print(f"\nSaving checkpoint to {checkpoint_path}...")
                    agent.save(checkpoint_path)

            pbar.close()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving final model...")
            pbar.close()
        except Exception as e:
            print(f"\nError during training: {e}")
            pbar.close()
            raise

        # Save final model
        final_path = os.path.join(save_path, "ppo_final.pt")
        print(f"Saving final model to {final_path}...")
        agent.save(final_path)

        # Generate and save training plots
        plots_dir = os.path.join(save_path, "plots")
        print(f"Generating training plots in {plots_dir}...")
        metrics.plot_metrics(plots_dir)

        print("Training completed successfully!")
        return agent

    except Exception as e:
        print(f"Error during training setup: {e}")
        raise


def evaluate(
    model_path: str,
    n_episodes: int = 100,
    opponent: str = "balanced",
    seed: Optional[int] = None,
    render: bool = False,
) -> None:
    """Evaluate trained PPO agent."""
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize opponent
    opponents = {
        "balanced": BalancedPlayer(),
        "adaptive": AdaptivePlayer(),
        "archer": ArcherPlayer(),
        "worker_rush": WorkerRushPlayer(),
        "army": ArmyPlayer(),
    }
    if opponent not in opponents:
        raise ValueError(f"Unknown opponent type: {opponent}")
    opponent_player = opponents[opponent]

    # Create environment
    env = StrategyGameEnv(opponent_player)

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0] if env.observation_space.shape else 0
    action_dim = env.action_space.shape[0] if env.action_space.shape else 0

    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )
    agent.load(model_path)

    # Initialize metrics tracker
    metrics = MetricsTracker()

    # Run evaluation episodes
    pbar = tqdm(range(n_episodes), desc="Evaluating", unit="episodes")
    for episode in pbar:
        obs, _ = env.reset(seed=seed)
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not (done or truncated):
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            action, _ = agent.policy.predict(obs_tensor)
            action = action.cpu().numpy()

            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += float(reward)  # Convert to float to avoid type error
            episode_length += 1

            if render:
                env.render()

        # Update metrics
        metrics.update(episode_reward, episode_length, info)

        # Update progress bar
        current_metrics = metrics.get_metrics()
        pbar.set_postfix(
            {
                "reward": f"{current_metrics['reward/mean']:.2f}",
                "win_rate": f"{current_metrics['win_rate']:.1%}",
            }
        )

    pbar.close()

    # Print final metrics
    final_metrics = metrics.get_metrics()
    print("\nFinal Evaluation Metrics:")
    print(
        f"Mean reward: {final_metrics['reward/mean']:.2f} ± {final_metrics['reward/std']:.2f}"
    )
    print(f"Mean episode length: {final_metrics['length/mean']:.1f}")
    print(f"Win rate: {final_metrics['win_rate']:.2%}")
    print(f"Mean success rate: {final_metrics['success_rate/mean']:.2%}")
    print(f"Mean episode time: {final_metrics['episode_time/mean']:.1f}s")
    print(f"Mean unit survival: {final_metrics['survival_rate/mean']:.2%}")
    print(f"Mean destruction ratio: {final_metrics['destruction_ratio/mean']:.2f}")
    print(f"Mean territory control: {final_metrics['territory_control/mean']:.2%}")
    print(f"Mean resource share: {final_metrics['resource_share/mean']:.2%}")

    # Generate and save evaluation plots
    plots_dir = os.path.join(os.path.dirname(model_path), "eval_plots")
    print(f"\nGenerating evaluation plots in {plots_dir}...")
    metrics.plot_metrics(plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-path", type=str, default="models")
    parser.add_argument("--opponent", type=str, default="balanced")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            seed=args.seed,
            device=args.device,
            log_interval=args.log_interval,
            save_path=args.save_path,
            opponent=args.opponent,
        )
    else:
        if args.model_path is None:
            raise ValueError("Must specify --model-path for evaluation")

        evaluate(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            opponent=args.opponent,
            seed=args.seed,
            render=args.render,
        )
