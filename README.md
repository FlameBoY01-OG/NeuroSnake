# 🐍 NeuroSnake - Deep Q-Learning Snake AI

Deep Q-Learning Snake agent with real-time neural network visualization and complete analytics.

## 🎯 Features

- **SimpleDQN** - 3-layer network (11→128→128→4)
- **GPU Accelerated** - CUDA training support
- **Live Visualization** - Neural network + Q-values + gameplay
- **Analytics Dashboard** - Evaluation + plots + heatmap
- **One Command** - Everything automated

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything (trains if needed, shows dashboard, launches visualization)
python main.py
```

That's it! One command does:

1. Trains model if not found (1000 episodes)
2. Shows complete analytics dashboard
3. Launches live visualization

Press ESC to exit visualization.

## 📋 Additional Commands

```bash
# Train only
python main.py train --episodes 2000

# Play only
python main.py play

# Dashboard only
python main.py dashboard

# Evaluate
python main.py eval --episodes 100
python main.py eval --compare  # Compare all checkpoints

# Analytics
python main.py plot     # Training plots
python main.py heatmap  # Exploration heatmap
python main.py record   # Record GIF

# Test setup
python main.py test
```

## 📊 Results

- Average Score: **29.04**
- Max Score: **54**
- Grid Coverage: **100%**
- Death Cause: 96% self-collision, 4% wall

## 🛠️ Technical Details

**Architecture**: SimpleDQN (11→128→128→4)
**State**: 11 features (danger, direction, food)
**Training**: 1000 episodes, Adam optimizer, epsilon decay
**Hardware**: CUDA GPU support

## 📜 License

MIT
