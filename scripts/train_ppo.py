"""Training script for PPO agent."""

import argparse
import os
from collections import deque
from typing import Dict, Optional, List, Any
import time

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
    """Tracks training metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.success_rates = []
        self.unit_survival_rates = []
        self.territory_controls = []
        self.units_created = []
        self.kills_vs_friendly = []
        self.gold_ratios = []
        self.gold_vs_opponent = []
        self.current_episode_reward = 0.0

    def update(self, reward: float, info: Dict[str, Any], done: bool) -> None:
        """Update metrics with new data."""
        self.current_episode_reward += reward

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
            
            # Track episode metrics
            self.episode_times.append(info.get("episode_time", 0.0))
            self.success_rates.append(info.get("success_rate", 0.0))
            self.unit_survival_rates.append(info.get("unit_survival_rate", 0.0))
            self.territory_controls.append(info.get("territory_control", 0.0))
            
            # Track unit metrics
            our_units = info.get("our_units", 0)
            enemy_units = info.get("enemy_units", 0)
            self.units_created.append(our_units)
            
            # Track gold metrics
            self.gold_ratios.append(info.get("gold_ratio", 0.0))
            self.gold_vs_opponent.append(info.get("gold_vs_opponent", 0.0))
            
            # Track combat metrics
            kills_ratio = our_units / (enemy_units + 1e-8)  # Avoid division by zero
            self.kills_vs_friendly.append(kills_ratio)

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        metrics = {}
        
        # Episode metrics
        if self.episode_rewards:
            metrics["mean_reward"] = float(np.mean(self.episode_rewards[-100:]))
            metrics["std_reward"] = float(np.std(self.episode_rewards[-100:]))
        
        # Success metrics
        if self.success_rates:
            metrics["success_rate"] = float(np.mean(self.success_rates[-100:]))
        
        # Time metrics
        if self.episode_times:
            metrics["mean_episode_time"] = float(np.mean(self.episode_times[-100:]))
        
        # Unit metrics
        if self.unit_survival_rates:
            metrics["survival_rate"] = float(np.mean(self.unit_survival_rates[-100:]))
        if self.units_created:
            metrics["units_created"] = float(np.mean(self.units_created[-100:]))
        if self.kills_vs_friendly:
            metrics["kills_friendly_ratio"] = float(np.mean(self.kills_vs_friendly[-100:]))
        
        # Territory metrics
        if self.territory_controls:
            metrics["territory_control"] = float(np.mean(self.territory_controls[-100:]))
        
        # Gold metrics
        if self.gold_ratios:
            metrics["gold_ratio"] = float(np.mean(self.gold_ratios[-100:]))
        if self.gold_vs_opponent:
            metrics["gold_vs_opponent"] = float(np.mean(self.gold_vs_opponent[-100:]))
        
        return metrics

    def plot_metrics(self, save_dir: str = "plots") -> None:
        """Generate and save plots of training metrics."""
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create plots directory
        os.makedirs(save_dir, exist_ok=True)

        # Set style
        sns.set_palette("husl")

        # Create a single large figure with all metrics
        fig = plt.figure(figsize=(20, 24))
        gs = plt.GridSpec(6, 2, figure=fig)
        fig.suptitle("Training Metrics Dashboard", fontsize=20, y=0.95)

        # Plot episode rewards
        ax1 = fig.add_subplot(gs[0, 0])
        if self.episode_rewards:
            sns.lineplot(x=range(len(self.episode_rewards)), y=self.episode_rewards, ax=ax1)
        ax1.set_title("Total Reward per Episode")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        # Plot success rate
        ax2 = fig.add_subplot(gs[0, 1])
        if self.success_rates:
            sns.lineplot(x=range(len(self.success_rates)), y=self.success_rates, ax=ax2)
        ax2.set_title("Success Rate")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Rate")
        ax2.set_ylim(0, 1)

        # Plot episode times
        ax3 = fig.add_subplot(gs[1, 0])
        if self.episode_times:
            sns.lineplot(x=range(len(self.episode_times)), y=self.episode_times, ax=ax3)
        ax3.set_title("Time to Win/Lose")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Time (s)")

        # Plot units created
        ax4 = fig.add_subplot(gs[1, 1])
        if self.units_created:
            sns.lineplot(x=range(len(self.units_created)), y=self.units_created, ax=ax4)
        ax4.set_title("Units Created")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Count")

        # Plot survival rate
        ax5 = fig.add_subplot(gs[2, 0])
        if self.unit_survival_rates:
            sns.lineplot(x=range(len(self.unit_survival_rates)), y=self.unit_survival_rates, ax=ax5)
        ax5.set_title("Unit Survival Rate")
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Rate")
        ax5.set_ylim(0, 1)

        # Plot kills vs friendly fire
        ax6 = fig.add_subplot(gs[2, 1])
        if self.kills_vs_friendly:
            sns.lineplot(x=range(len(self.kills_vs_friendly)), y=self.kills_vs_friendly, ax=ax6)
        ax6.set_title("Kills vs Friendly Fire Ratio")
        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Ratio")

        # Plot territory control
        ax7 = fig.add_subplot(gs[3, 0])
        if self.territory_controls:
            sns.lineplot(x=range(len(self.territory_controls)), y=self.territory_controls, ax=ax7)
        ax7.set_title("Territory Control")
        ax7.set_xlabel("Episode")
        ax7.set_ylabel("Control %")
        ax7.set_ylim(0, 1)

        # Plot gold ratio
        ax8 = fig.add_subplot(gs[3, 1])
        if self.gold_ratios:
            sns.lineplot(x=range(len(self.gold_ratios)), y=self.gold_ratios, ax=ax8)
        ax8.set_title("Gold Ratio")
        ax8.set_xlabel("Episode")
        ax8.set_ylabel("Ratio")

        # Plot gold vs opponent
        ax9 = fig.add_subplot(gs[4, :])
        if self.gold_vs_opponent:
            sns.lineplot(x=range(len(self.gold_vs_opponent)), y=self.gold_vs_opponent, ax=ax9)
        ax9.set_title("Gold vs Opponent")
        ax9.set_xlabel("Episode")
        ax9.set_ylabel("Ratio")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_dashboard.png"), dpi=300, bbox_inches='tight')
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
    device: str = "cuda",
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
            device=device
        )
        print(f"Agent initialized with state_dim={state_dim}, action_dim={action_dim}, device={device}")

        # Initialize metrics tracker
        print("Initializing metrics tracker...")
        metrics = MetricsTracker()

        # Training loop
        print("Starting training loop...")
        timesteps_elapsed = 0
        episodes_completed = 0
        try:
            # Create progress bar for total training
            pbar = tqdm(
                total=total_timesteps,
                desc="Training Progress",
                unit="steps",
                position=0,
                leave=True,
            )

            while timesteps_elapsed < total_timesteps and episodes_completed < 1000:  # Stop at 1000 episodes
                # Collect rollout
                rollout_metrics = agent.collect_rollouts(env)
                timesteps_elapsed += agent.config.n_steps
                episodes_completed += len(rollout_metrics.get("episode_rewards", []))
                pbar.update(agent.config.n_steps)

                # Print episode progress
                print(f"\nEpisodes completed: {episodes_completed}/1000")

                # Compute advantages and returns
                with torch.no_grad():
                    advantages = agent._compute_advantages()
                    returns = advantages + agent.value_buffer.squeeze(-1)
                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Print shapes for debugging
                print(f"\nAdvantages shape: {advantages.shape}")
                print(f"Returns shape: {returns.shape}")
                print(f"Value buffer shape: {agent.value_buffer.shape}")

                # Train on collected experience
                train_metrics = agent.train_epoch(
                    obs=agent.obs_buffer,
                    actions=agent.action_buffer,
                    old_log_probs=agent.log_prob_buffer,
                    advantages=advantages.detach(),
                    returns=returns.detach()
                )

                # Update metrics
                metrics.update(
                    reward=rollout_metrics["mean_reward"],
                    info=rollout_metrics,
                    done=True
                )

                # Log metrics
                if timesteps_elapsed % (log_interval * agent.config.n_steps) == 0:
                    current_metrics = metrics.get_metrics()
                    pbar.set_postfix(
                        {
                            "reward": f"{current_metrics['mean_reward']:.2f}±{current_metrics['std_reward']:.2f}",
                            "success_rate": f"{current_metrics['success_rate']:.1%}",
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
            episode_reward += float(reward)
            episode_length += 1

            if render:
                env.render()

            # Update metrics if episode is done
            if done or truncated:
                metrics.update(
                    reward=episode_reward,
                    info=info,
                    done=True
                )
                break

        # Update progress bar
        current_metrics = metrics.get_metrics()
        pbar.set_postfix(
            {
                "reward": f"{current_metrics['mean_reward']:.2f}",
                "success_rate": f"{current_metrics['success_rate']:.1%}",
            }
        )

    pbar.close()

    # Print final metrics
    final_metrics = metrics.get_metrics()
    print("\nFinal Evaluation Metrics:")
    print(
        f"Mean reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}"
    )
    print(f"Mean episode length: {final_metrics['mean_episode_time']:.1f}")
    print(f"Success rate: {final_metrics['success_rate']:.2%}")
    print(f"Mean unit survival: {final_metrics['survival_rate']:.2%}")
    print(f"Mean destruction ratio: {final_metrics['kills_friendly_ratio']:.2f}")
    print(f"Mean territory control: {final_metrics['territory_control']:.2%}")
    print(f"Mean resource share: {final_metrics['gold_ratio']:.2%}")

    # Generate and save evaluation plots
    plots_dir = os.path.join(os.path.dirname(model_path), "eval_plots")
    print(f"\nGenerating evaluation plots in {plots_dir}...")
    metrics.plot_metrics(plots_dir)


def evaluate_against_multiple_opponents(
    model_path: str,
    n_rounds: int = 400,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> None:
    """Evaluate trained PPO agent against multiple opponents.
    
    Args:
        model_path: Path to the trained model
        n_rounds: Number of rounds to play against each opponent
        seed: Random seed for reproducibility
        device: Device to run evaluation on
    """
    print(f"\nEvaluating model from {model_path}")
    print(f"Playing {n_rounds} rounds against each opponent")
    
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize all opponents
    opponents = {
        "Balanced": BalancedPlayer(),
        "Adaptive": AdaptivePlayer(),
        "Archer": ArcherPlayer(),
        "Worker Rush": WorkerRushPlayer(),
        "Army": ArmyPlayer(),
    }

    # Load the trained agent
    env = StrategyGameEnv(BalancedPlayer())  # Temporary env for initialization
    state_dim = env.observation_space.shape[0] if env.observation_space.shape else 0
    action_dim = env.action_space.shape[0] if env.action_space.shape else 0
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    agent.load(model_path)
    
    # Results storage
    results = []
    
    # Evaluate against each opponent
    for opponent_name, opponent in opponents.items():
        print(f"\nEvaluating against {opponent_name}...")
        
        # Create environment with current opponent
        env = StrategyGameEnv(opponent)
        
        # Initialize metrics tracker for this opponent
        metrics = MetricsTracker()
        
        # Statistics for this opponent
        ppo_wins = 0
        ppo_total_reward = 0
        opponent_wins = 0
        draws = 0
        episode_lengths = []
        win_times = []
        
        # Progress bar for rounds
        pbar = tqdm(range(n_rounds), desc=f"Playing vs {opponent_name}")
        
        for episode in pbar:
            obs, _ = env.reset(seed=seed)
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            episode_start_time = time.time()
            
            while not (done or truncated):
                # Get action from policy
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                action, _ = agent.policy.predict(obs_tensor)
                action = action.cpu().numpy().squeeze()  # Remove batch dimension
                
                # Take step in environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += float(reward)
                episode_length += 1

                # Update metrics if episode is done
                if done or truncated:
                    metrics.update(
                        reward=episode_reward,
                        info=info,
                        done=True
                    )
                    break
            
            # Record episode statistics
            episode_lengths.append(episode_length)
            episode_time = time.time() - episode_start_time
            
            if info["our_units"] > info["enemy_units"]:
                ppo_wins += 1
                win_times.append(episode_time)
            elif info["our_units"] < info["enemy_units"]:
                opponent_wins += 1
            else:
                draws += 1
            
            ppo_total_reward += episode_reward
            
            # Update progress bar
            win_rate = (ppo_wins / (episode + 1)) * 100
            pbar.set_postfix({
                "PPO Wins": f"{win_rate:.1f}%",
                "Avg Length": f"{np.mean(episode_lengths):.1f}"
            })
        
        # Compute statistics
        result = {
            "opponent": opponent_name,
            "ppo_wins": ppo_wins,
            "opponent_wins": opponent_wins,
            "draws": draws,
            "win_rate": (ppo_wins / n_rounds) * 100,
            "avg_reward": ppo_total_reward / n_rounds,
            "avg_episode_length": np.mean(episode_lengths),
            "avg_win_time": np.mean(win_times) if win_times else float('inf'),
            "total_episodes": n_rounds
        }
        results.append(result)
        
        # Print detailed results for this opponent
        print(f"\nResults vs {opponent_name}:")
        print(f"PPO Wins: {ppo_wins} ({result['win_rate']:.1f}%)")
        print(f"Opponent Wins: {opponent_wins} ({(opponent_wins/n_rounds)*100:.1f}%)")
        print(f"Draws: {draws} ({(draws/n_rounds)*100:.1f}%)")
        print(f"Average Episode Length: {result['avg_episode_length']:.1f}")
        print(f"Average Win Time: {result['avg_win_time']:.1f}s")
        print(f"Average Reward: {result['avg_reward']:.2f}")
    
    # Print summary table
    print("\nOverall Results Summary:")
    print("-" * 80)
    print(f"{'Opponent':<15} {'Win Rate':<10} {'Avg Reward':<12} {'Avg Length':<12} {'Avg Win Time':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['opponent']:<15} {r['win_rate']:>8.1f}% {r['avg_reward']:>10.2f} {r['avg_episode_length']:>10.1f} {r['avg_win_time']:>10.1f}s")
    print("-" * 80)
    
    # Plot results
    plot_evaluation_results(results, save_dir=os.path.dirname(model_path))


def plot_evaluation_results(results: List[Dict], save_dir: str) -> None:
    """Plot evaluation results.
    
    Args:
        results: List of result dictionaries
        save_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use("seaborn")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("PPO Agent Evaluation Results", fontsize=16)
    
    # Prepare data
    opponents = [r["opponent"] for r in results]
    win_rates = [r["win_rate"] for r in results]
    avg_rewards = [r["avg_reward"] for r in results]
    avg_lengths = [r["avg_episode_length"] for r in results]
    avg_win_times = [r["avg_win_time"] for r in results]
    
    # Win Rate plot
    sns.barplot(x=opponents, y=win_rates, ax=ax1)
    ax1.set_title("Win Rate by Opponent")
    ax1.set_ylabel("Win Rate (%)")
    ax1.tick_params(axis='x', rotation=45)
    
    # Average Reward plot
    sns.barplot(x=opponents, y=avg_rewards, ax=ax2)
    ax2.set_title("Average Reward by Opponent")
    ax2.set_ylabel("Average Reward")
    ax2.tick_params(axis='x', rotation=45)
    
    # Average Episode Length plot
    sns.barplot(x=opponents, y=avg_lengths, ax=ax3)
    ax3.set_title("Average Episode Length by Opponent")
    ax3.set_ylabel("Steps")
    ax3.tick_params(axis='x', rotation=45)
    
    # Average Win Time plot
    sns.barplot(x=opponents, y=avg_win_times, ax=ax4)
    ax4.set_title("Average Win Time by Opponent")
    ax4.set_ylabel("Time (s)")
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "evaluation_results.png"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "eval_multi"])
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
    parser.add_argument("--n-rounds", type=int, default=400)
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
    elif args.mode == "eval":
        if args.model_path is None:
            raise ValueError("Must specify --model-path for evaluation")

        evaluate(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            opponent=args.opponent,
            seed=args.seed,
            render=args.render,
        )
    else:  # eval_multi
        if args.model_path is None:
            raise ValueError("Must specify --model-path for evaluation")
            
        evaluate_against_multiple_opponents(
            model_path=args.model_path,
            n_rounds=args.n_rounds,
            seed=args.seed,
            device=args.device
        )
