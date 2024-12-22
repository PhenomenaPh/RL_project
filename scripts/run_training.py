"""Script for running training with flexible configuration."""

import os
import sys
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agents.ppo_agent import PPOAgent, PPOConfig
from scripts.train_ppo import train


def run_training(
    # Training parameters
    experiment_name: str = "default",
    total_timesteps: int = 1_000_000,
    save_path: str = "models",
    opponent: str = "balanced",
    seed: int | None = None,
    device: str = "auto",
    # PPO hyperparameters
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
    target_kl: float = 0.015,
    # Logging parameters
    log_interval: int = 1,
) -> PPOAgent:
    """Run a training experiment with the specified configuration.

    Args:
        experiment_name: Name of the experiment (used for saving)
        total_timesteps: Total number of timesteps to train for
        save_path: Base directory to save models and plots
        opponent: Type of opponent to train against
        seed: Random seed for reproducibility
        device: Device to run on ('auto', 'cuda', or 'cpu')
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to run for each environment per update
        batch_size: Minibatch size for updates
        n_epochs: Number of epoch when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for GAE
        clip_range: Clipping parameter for PPO
        ent_coef: Entropy coefficient for loss calculation
        vf_coef: Value function coefficient for loss calculation
        max_grad_norm: Max norm of gradients for gradient clipping
        target_kl: Target KL divergence threshold for early stopping
        log_interval: Number of timesteps between logging events
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_path, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save configuration
    config = {
        "experiment_name": experiment_name,
        "total_timesteps": total_timesteps,
        "opponent": opponent,
        "seed": seed,
        "device": device,
        "ppo_config": {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "target_kl": target_kl,
        },
        "log_interval": log_interval,
    }

    # Train agent
    trained_agent = train(
        total_timesteps=total_timesteps,
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
        seed=seed,
        device=device,
        log_interval=log_interval,
        save_path=experiment_dir,
        opponent=opponent,
    )

    return trained_agent


if __name__ == "__main__":
    # Example usage:

    # 1. Basic training with default parameters
    agent = run_training(
        experiment_name="basic_training",
        total_timesteps=100_000,  # Reduced for example
    )

    # 2. Training against different opponents
    opponents = ["balanced", "adaptive", "archer", "worker_rush", "army"]
    for opponent in opponents:
        agent = run_training(
            experiment_name=f"opponent_{opponent}",
            total_timesteps=100_000,  # Reduced for example
            opponent=opponent,
            seed=42,  # For reproducibility
        )

    # 3. Hyperparameter experimentation
    learning_rates = [1e-4, 3e-4, 1e-3]
    for lr in learning_rates:
        agent = run_training(
            experiment_name="lr_sweep",
            total_timesteps=100_000,  # Reduced for example
            learning_rate=lr,
            seed=42,  # For reproducibility
        )
