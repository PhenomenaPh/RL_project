# Strategy Game RL Project

This project implements a reinforcement learning agent using Proximal Policy Optimization (PPO) to play a strategy game. The agent learns to make tactical decisions in a game environment where it competes against various opponent types.

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

### Evaluation

To evaluate a trained agent:

```bash
python scripts/train_ppo.py --mode eval \
    --model-path models/ppo_final.pt \
    --opponent balanced \
    --n-episodes 100
```

## Components

- `src/agents/ppo_agent.py`: Implementation of the PPO agent
- `src/environment/game_env.py`: Game environment implementation
- `src/environment/mini_strat_game.py`: Core game mechanics and rules
- `scripts/train_ppo.py`: Training and evaluation script

## License

MIT License 