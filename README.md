# Strategy Game RL Project

This project implements a reinforcement learning agent using Proximal Policy Optimization (PPO) to play a strategy game. The agent learns to make tactical decisions in a game environment where it competes against various opponent types.

## Game Mechanics

### Overview
The game is a real-time strategy game where players compete for resources and military dominance. Players can build different types of units and must manage their economy while maintaining a military presence.

### Units
- **Worker**: Gathers resources and builds basic structures
- **Pikeman**: Strong against cavalry units
- **Swordsman**: Balanced melee unit
- **Archer**: Ranged unit, effective against infantry
- **Medic**: Heals friendly units
- **Knight**: Fast and powerful cavalry unit

### Win Conditions
Victory can be achieved through:
1. **Economic Dominance**: Having significantly more gold (2x) than the opponent while maintaining military superiority
2. **Complete Victory**: Eliminating all enemy units while having more gold
3. **Time Limit**: Reaching the maximum number of steps with better economic and military position

## Project Structure

```
.
├── src/                    # Source code
│   ├── agents/            # Agent implementations
│   ├── environment/       # Game environment
│   └── utils/            # Utility functions
├── scripts/               # Training and evaluation scripts
├── models/               # Saved models
├── tests/                # Test files
└── README.md             # Project documentation
```

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a new PPO agent:

```bash
python scripts/train_ppo.py --mode train \
    --total-timesteps 100000 \
    --opponent balanced \
    --log-interval 1 \
    --save-path models
```

Key training parameters:
- `--total-timesteps`: Total number of environment steps for training
- `--opponent`: Type of opponent (balanced, adaptive, archer, worker_rush, army)
- `--device`: Training device (auto, cuda, cpu)
- `--batch-size`: Batch size for training updates
- `--learning-rate`: Learning rate for the optimizer

### Evaluation

To evaluate a trained agent:

```bash
python scripts/train_ppo.py --mode eval \
    --model-path models/ppo_final.pt \
    --opponent balanced \
    --n-episodes 100 \
    --render
```

Evaluation options:
- `--model-path`: Path to the trained model
- `--n-episodes`: Number of evaluation episodes
- `--render`: Enable visualization of the game
- `--opponent`: Type of opponent to evaluate against

## Components

- `src/agents/ppo_agent.py`: Implementation of the PPO agent with CUDA support
- `src/environment/game_env.py`: Game environment with custom reward shaping
- `src/environment/mini_strat_game.py`: Core game mechanics and rules
- `scripts/train_ppo.py`: Training and evaluation script with metrics tracking

## Training Metrics

The training process tracks several metrics:
- Episode rewards and win rates
- Resource collection efficiency
- Unit survival rates
- Territory control
- Combat effectiveness (destruction ratio)
- Training loss and performance improvements

All metrics are automatically plotted and saved in the models directory.

## License

MIT License 