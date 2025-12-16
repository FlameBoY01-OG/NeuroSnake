import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_training_results(log_file="training_log.txt", output_dir="plots"):
    """Generate plots from training log."""
    os.makedirs(output_dir, exist_ok=True)
    
    episodes = []
    rewards = []
    scores = []
    losses = []
    
    # Parse log file
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'Ep:' in line:
                    parts = line.split('|')
                    ep = int(parts[0].split(':')[1].strip())
                    reward = float(parts[1].split(':')[1].strip())
                    score = int(parts[3].split(':')[1].strip())
                    loss = float(parts[6].split(':')[1].strip())
                    
                    episodes.append(ep)
                    rewards.append(reward)
                    scores.append(score)
                    losses.append(loss)
    except FileNotFoundError:
        print(f"Error: {log_file} not found. Run training first.")
        return
    
    if not episodes:
        print("No training data found in log file.")
        return
    
    # Style
    plt.style.use('dark_background')
    
    # Plot 1: Score over episodes
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, scores, alpha=0.3, color='cyan', linewidth=0.5)
    window = 50
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, color='lime', linewidth=2, label=f'{window}-episode moving average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score vs Episode', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_vs_episode.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'score_vs_episode.png')}")
    plt.close()
    
    # Plot 2: Reward over episodes
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, rewards, alpha=0.3, color='orange', linewidth=0.5)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'{window}-episode moving average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward vs Episode', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_vs_episode.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'reward_vs_episode.png')}")
    plt.close()
    
    # Plot 3: Loss over episodes
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, losses, alpha=0.3, color='magenta', linewidth=0.5)
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, color='yellow', linewidth=2, label=f'{window}-episode moving average')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss vs Episode', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_vs_episode.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'loss_vs_episode.png')}")
    plt.close()
    
    # Plot 4: Combined dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Score
    axes[0, 0].plot(episodes, scores, alpha=0.3, color='cyan', linewidth=0.5)
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], moving_avg, color='lime', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Score Progress')
    axes[0, 0].grid(alpha=0.2)
    
    # Reward
    axes[0, 1].plot(episodes, rewards, alpha=0.3, color='orange', linewidth=0.5)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], moving_avg, color='red', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Reward Progress')
    axes[0, 1].grid(alpha=0.2)
    
    # Loss
    axes[1, 0].plot(episodes, losses, alpha=0.3, color='magenta', linewidth=0.5)
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(episodes[window-1:], moving_avg, color='yellow', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].grid(alpha=0.2)
    
    # Score distribution
    axes[1, 1].hist(scores, bins=30, color='cyan', alpha=0.7, edgecolor='white')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].grid(alpha=0.2)
    
    plt.suptitle('NeuroSnake Training Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_dashboard.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'training_dashboard.png')}")
    plt.close()
    
    print(f"\n✓ All plots generated in '{output_dir}/' directory")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate training plots")
    parser.add_argument("--log", type=str, default="training_log.txt", help="Training log file")
    parser.add_argument("--output", type=str, default="plots", help="Output directory")
    args = parser.parse_args()
    
    plot_training_results(args.log, args.output)
