# SpelunkyRL Examples

This directory contains example scripts demonstrating how to use SpelunkyRL for reinforcement learning.

## Examples Overview

| Script | Purpose |
|--------|---------|
| `manual_control.py` | Test environment with keyboard control |
| `train_get_to_exit.py` | Train an agent from scratch |
| `evaluate_model.py` | Evaluate a trained model |
| `record_video.py` | Record video of trained agent |

## Quick Start

### 1. Manual Control (Test Your Installation)

Test your SpelunkyRL installation by controlling the character manually:

```bash
python examples/manual_control.py
```

**What it does:**
- Opens Spelunky 2 with keyboard control enabled
- God mode activated (invulnerability)
- Good for testing your setup and understanding game mechanics

### 2. Train an Agent

Train an RL agent to navigate to the level exit, using SB3:

```bash
python examples/train_get_to_exit.py
```

**What it does:**
- Trains a RecurrentPPO agent with LSTM
- Uses 6 parallel environments for faster training
- Saves checkpoints every 5 rollouts
- Logs training metrics to TensorBoard

**Monitor progress:**
```bash
tensorboard --logdir=./tensorboard_logs
```

### 3. Evaluate a Trained Model

Test a trained model's performance:

```bash
python examples/evaluate_model.py
```

**Before running:**
- Update `MODEL_PATH` in the script to point to your trained model
- Example: `MODEL_PATH = "./models_get_to_exit/final_model.zip"`

**What it does:**
- Runs 50 evaluation episodes
- Calculates success rate and statistics
- Reports average completion time for successful episodes

### 4. Record Video of Agent

Create a video of your trained agent playing:

```bash
python examples/record_video.py
```

**Before running:**
- Update `MODEL_PATH` in the script
- Install opencv: `pip install opencv-python`

**What it does:**
- Records 30 seconds of gameplay
- Saves as MP4 in `./videos/` directory
- Shows agent's learned behavior

## Configuration

All scripts require updating these paths to match your installation:

```python
spelunky_dir = r"C:\Path\To\Your\Spelunky 2"
playlunky_dir = r"C:\Path\To\Your\modlunky2\playlunky\nightly"
```

Common locations:
- **Spelunky 2:** Steam library folder (e.g., `C:\Program Files (x86)\Steam\steamapps\common\Spelunky 2`)
- **Playlunky:** Usually in `%LOCALAPPDATA%\spelunky.fyi\modlunky2\playlunky\nightly`

## Requirements

### Basic Installation

For just using the environments (no training):
```bash
pip install spelunkyRL
```

### For Training (recommended)

To run the training examples:
```bash
pip install spelunkyRL[train]
```

This installs: `torch`, `stable-baselines3`, `sb3-contrib`

**Note:** GPU (CUDA) is highly recommended for training. Training on CPU is possible but much slower (10+ hours vs 2-4 hours).

### For Video Recording

To record videos of trained agents:
```bash
pip install spelunkyRL[video]
```

This installs: `opencv-python`

### Everything

To install all optional dependencies:
```bash
pip install spelunkyRL[all]
```

## Customization

### Changing the Environment

Replace `SpelunkyEnv` import to try different tasks:

```python
# from spelunkyRL.environments.get_to_exit import SpelunkyEnv
from spelunkyRL.environments.gold_grabber import SpelunkyEnv  # Collect gold
from spelunkyRL.environments.enemy_killer import SpelunkyEnv   # Kill enemies
```

### Adjusting Hyperparameters

In `train_get_to_exit.py`, you can modify:

```python
NUM_ENVS = 6              # Parallel environments (more = faster)
TIMESTEPS_PER_ROLLOUT = 2048  # Steps before model update
learning_rate = 3e-4      # Learning rate
lstm_hidden_size = 64     # LSTM size
```

### Network Architecture

Modify `SpelunkyFeaturesExtractor` to experiment with:
- Different CNN architectures
- Embedding dimensions
- Feature fusion strategies

## Next Steps

After running these examples:

1. **Create custom environments:** See `spelunkyRL/environments/template_environment.py`
2. **Design custom reward functions:** Modify reward shaping for your task
3. **Experiment with algorithms:** Try different RL algorithms from stable-baselines3
4. **Multi-task learning:** Train agents on multiple objectives