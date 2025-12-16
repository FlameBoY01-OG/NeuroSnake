# 🐍 NeuroSnake - Deep Q-Learning Snake AI

A professional Deep Q-Learning implementation for Snake with real-time neural network visualization and explainable AI features.

## 🎯 Features

- **SimpleDQN Architecture** - 3-layer network (11→128→128→4) optimized for Snake
- **11-Feature State** - Clean representation: danger detection, direction, food location
- **GPU Training** - CUDA-accelerated learning
- **Complete Visualization** - Game, neural network, and Q-values in one window
- **Training Analytics** - Score/reward/loss plots and comprehensive statistics
- **Death Analysis** - Tracks wall collisions vs self-collisions

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

The project includes a unified CLI interface:

```bash
# Train a new model
python main.py train
python main.py train --episodes 2000  # Custom episode count

# Watch the trained agent play with visualization
python main.py play
python main.py play --model model_checkpoints/policy_ep800.pth  # Specific checkpoint

# Evaluate model performance
python main.py eval
python main.py eval --episodes 200  # More episodes for better stats
python main.py eval --compare  # Compare all checkpoints

# Test environment setup
python main.py test

# Generate training plots (after training)
python main.py plot

# Record gameplay as GIF
python main.py record --episodes 3
```

## 📊 Performance

**Best Checkpoint**: `policy_ep300.pth`

- Average Score: **38.33**
- Max Score: **54**
- Average Steps: **446.67**

**Final Model**: `policy_final.pth` (Episode 1000)

- Average Score: **27.80**
- Death Causes: **100% self-collision** (excellent wall avoidance!)

**Known Issue**: Model has perfect wall avoidance but dies exclusively from self-collisions. Future work should improve body awareness.

## 🎮 Visualization Features

The complete view shows:

- **Game Board**: Snake, food, grid (left side)
- **Neural Network**: All 4 layers with animated connections (right side)
- **Q-Values**: Bar chart showing decision-making (bottom right)
- **Stats**: Score, length, FPS, current action (bottom left)

Press **ESC** to exit visualization.

## 🔧 Configuration

Edit values in files directly:

- Grid size: `env/snake_env.py` (default: 20×20)
- Network size: `model/dqn.py` (default: 128 hidden units)
- Training params: `training/train.py` (episodes, lr, batch_size)
- Visualization: `visualization/complete_view.py` (colors, layout)

## 📝 Training Tips

1. **Monitor early performance** - Good models show progress by episode 200-400
2. **Try different learning rates** - 1e-3 works well, try 5e-4 for stability
3. **Watch the plots** - Steady score increase = good learning
4. **Death analysis matters** - 100% self-collision means body awareness needs work

## 🐛 Troubleshooting

**ModuleNotFoundError: No module named 'torch'**

```bash
# Activate conda environment first
conda activate base
python main.py
```

**CUDA not available**

- Works fine on CPU (just slower training)
- For GPU: Install PyTorch with CUDA support

**No plots showing**

- Training log doesn't exist - run training first
- Plots generate after: `python main.py train`

## 🎯 Future Improvements

- [ ] Add heatmap visualization for explored states
- [ ] Implement curriculum learning (start with smaller grid)
- [ ] Try Dueling DQN or Rainbow DQN
- [ ] Add MP4 video recording (currently GIF only)
- [ ] Improve body awareness (reduce self-collision deaths)

## 📜 License

MIT License - Feel free to use for learning and projects!

## 🙏 Acknowledgments

Built with PyTorch, Pygame, and love for reinforcement learning.

---

**Made with ❤️ for AI enthusiasts**
