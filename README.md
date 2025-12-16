# 🐍 NeuroSnake - Deep Q-Learning Snake AI

> **⚠️ WORK IN PROGRESS** - Active development and improvements ongoing

A professional implementation of Deep Q-Learning (DQN) applied to the classic Snake game, featuring real-time neural network visualization and GPU-accelerated training.

## 🎯 Features

- **Simple but Effective DQN Architecture** - 3-layer network (128→128→4) optimized for Snake gameplay
- **Clean State Representation** - 11 features capturing danger, direction, and food location
- **GPU Training** - CUDA-accelerated for fast learning (20-30 minutes for 1000 episodes)
- **Real-time Visualization** - Watch the neural network think as the snake plays
- **Catppuccin-themed UI** - Beautiful, research-grade visualization interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Pygame

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuroSnake.git
cd NeuroSnake

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train a new model (1000 episodes, ~20-30 mins on GPU)
python -m training.train
```

### Play & Visualize

```bash
# Watch the trained agent play with neural network visualization
python -m visualization.render
```

## 📊 Performance

- **Episode 200**: Agent starts eating food consistently
- **Episode 500**: Average score of 1-2
- **Episode 800**: Average score of 5-8
- **Episode 1000**: Average score of 10-12 (professional play level)

## 🧠 Architecture

### Neural Network

```
Input (11 features) → Hidden (128) → Hidden (128) → Output (4 actions)
                         ReLU            ReLU
```

### State Features (11 total)

- **Danger Detection (3)**: Immediate collision detection in straight, left, right directions
- **Current Direction (4)**: One-hot encoding of snake's heading
- **Food Location (4)**: Binary indicators for food in each cardinal direction

### Reward Structure

- **+10**: Eating food
- **-10**: Collision (death)
- **0**: Normal move

## 🛠️ Project Structure

```
NeuroSnake/
├── env/
│   ├── __init__.py
│   └── snake_env.py          # Snake game environment
├── model/
│   ├── __init__.py
│   ├── agent.py               # DQN Agent implementation
│   ├── dqn.py                 # Neural network architecture
│   └── replay_memory.py       # Experience replay buffer
├── training/
│   ├── train.py               # Training loop
│   └── evaluate.py            # Model evaluation
├── visualization/
│   ├── __init__.py
│   ├── render.py              # Main visualization
│   ├── render_new.py          # Enhanced visualizer
│   ├── activations.py         # Neural activation display
│   └── q_panel.py             # Q-value display panel
├── model_checkpoints/         # Saved models
├── assets/                    # Game assets
└── requirements.txt
```

## 📈 Training Details

- **Algorithm**: Deep Q-Network (DQN)
- **Learning Rate**: 1e-3
- **Batch Size**: 64
- **Discount Factor (γ)**: 0.99
- **Epsilon Decay**: Linear from 1.0 to 0.01 over 80k steps
- **Target Network Update**: Every 1000 steps
- **Memory Size**: 50,000 transitions

## 🎮 Controls

During visualization:

- **ESC**: Quit
- Watch the AI play automatically

## 🔧 Configuration

Edit hyperparameters in [training/train.py](training/train.py):

- `num_episodes`: Number of training episodes
- `lr`: Learning rate
- `batch_size`: Mini-batch size
- `gamma`: Discount factor

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! This is an active work-in-progress project.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Status**: 🚧 Active Development - Model training successfully, achieving professional-level play
