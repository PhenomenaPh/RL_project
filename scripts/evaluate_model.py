"""Script for evaluating a trained model against all opponents."""

import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scripts.train_ppo import evaluate_against_multiple_opponents

if __name__ == "__main__":
    model_path = "models/improved_training_latest/ppo_final.pt"
    
    print(f"\nEvaluating model from {model_path}")
    evaluate_against_multiple_opponents(
        model_path=model_path,
        n_rounds=100,  # Number of rounds to play against each opponent
        seed=42,
        device="cuda"  # Will automatically use CPU if CUDA not available
    ) 