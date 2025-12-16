import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from env.snake_env import SnakeEnv
from model.agent import DQNAgent

def generate_heatmap(model_path, episodes=100, output_path="plots/heatmap.png"):
    """
    Generate heatmap showing which grid cells the snake visited most frequently.
    Helps visualize exploration patterns and identify unexplored areas.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = SnakeEnv(width=20, height=20)
    state_dim = env.reset().shape[0]
    
    agent = DQNAgent(
        input_dim=state_dim,
        output_dim=4,
        device=device,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        memory_capacity=50000,
        use_prioritized_replay=False,
        target_update_steps=1000
    )
    
    agent.load(model_path)
    print(f"Loaded model: {model_path}")
    print(f"Generating heatmap over {episodes} episodes...")
    
    # Track visit counts for each cell
    visit_counts = np.zeros((env.height, env.width))
    
    for ep in range(episodes):
        state = env.reset(randomize_start=True)
        max_steps = 1000
        
        for step in range(max_steps):
            # Track head position
            visit_counts[env.head.y, env.head.x] += 1
            
            # Take action
            action = agent.select_action(state, eps=0.0)
            next_state, reward, done = env.step(action)
            state = next_state
            
            if done:
                break
    
    # Create the heatmap
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Normalize and plot
    im = ax.imshow(visit_counts, cmap='hot', interpolation='nearest')
    ax.set_title(f'Snake Exploration Heatmap ({episodes} episodes)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visit Count', rotation=270, labelpad=20, fontsize=12)
    
    # Add grid
    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add statistics
    total_visits = visit_counts.sum()
    explored_cells = np.count_nonzero(visit_counts)
    total_cells = env.width * env.height
    coverage = (explored_cells / total_cells) * 100
    
    stats_text = f'Coverage: {coverage:.1f}% ({explored_cells}/{total_cells} cells)\n'
    stats_text += f'Total visits: {int(total_visits)}\n'
    stats_text += f'Max visits: {int(visit_counts.max())}'
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Heatmap saved: {output_path}")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Explored cells: {explored_cells}/{total_cells}")
    
    plt.close()
    return visit_counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate exploration heatmap")
    parser.add_argument("--model", type=str, default="model_checkpoints/policy_final.pth")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="plots/heatmap.png")
    args = parser.parse_args()
    
    generate_heatmap(args.model, args.episodes, args.output)
