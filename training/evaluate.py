# training/evaluate.py
import numpy as np
import torch
import sys
import os
from env.snake_env import SnakeEnv
from model.agent import DQNAgent

def evaluate_model(model_path, num_episodes=100, render=False, device=None):
    """
    Evaluate a trained model over multiple episodes.
    
    Args:
        model_path: Path to saved model checkpoint
        num_episodes: Number of episodes to evaluate
        render: Whether to render the game (slow)
        device: PyTorch device
    
    Returns:
        dict: Statistics including avg_score, max_score, avg_steps, etc.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # Initialize environment and agent
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
    
    # Load model
    agent.load(model_path)
    print(f"Loaded model: {model_path}")
    print(f"Device: {device}")
    print(f"Evaluating over {num_episodes} episodes...\n")
    
    # Collect statistics
    scores = []
    steps_list = []
    rewards_list = []
    death_causes = {'wall': 0, 'self': 0}
    
    for ep in range(1, num_episodes + 1):
        state = env.reset(randomize_start=True)
        ep_reward = 0
        ep_steps = 0
        max_steps = 1000  # Limit to prevent infinite loops
        
        while ep_steps < max_steps:
            # Greedy action (no exploration)
            action = agent.select_action(state, eps=0.0)
            next_state, reward, done = env.step(action)
            
            ep_reward += reward
            ep_steps += 1
            state = next_state
            
            if done:
                # Determine death cause
                head = env.head
                if head.x < 0 or head.x >= env.width or head.y < 0 or head.y >= env.height:
                    death_causes['wall'] += 1
                else:
                    death_causes['self'] += 1
                break
        
        scores.append(env.score)
        steps_list.append(ep_steps)
        rewards_list.append(ep_reward)
        
        if ep % 5 == 0 or ep == num_episodes:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            avg_steps = np.mean(steps_list[-10:]) if len(steps_list) >= 10 else np.mean(steps_list)
            print(f"Ep {ep:3d}/{num_episodes} | Score: {env.score:2d} | Avg: {avg_score:.1f} | Steps: {ep_steps:4d}")
    
    # Calculate statistics
    stats = {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'median_score': np.median(scores),
        'avg_steps': np.mean(steps_list),
        'avg_reward': np.mean(rewards_list),
        'scores': scores,
        'steps': steps_list,
        'rewards': rewards_list,
        'death_causes': death_causes,
    }
    
    return stats

def print_statistics(stats):
    """Print formatted evaluation statistics."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Average Score:    {stats['avg_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"Median Score:     {stats['median_score']:.1f}")
    print(f"Max Score:        {stats['max_score']}")
    print(f"Min Score:        {stats['min_score']}")
    print(f"Average Steps:    {stats['avg_steps']:.1f}")
    print(f"Average Reward:   {stats['avg_reward']:.2f}")
    print(f"\nDeath Causes:")
    print(f"  Wall collision:  {stats['death_causes']['wall']} ({stats['death_causes']['wall']/len(stats['scores'])*100:.1f}%)")
    print(f"  Self collision:  {stats['death_causes']['self']} ({stats['death_causes']['self']/len(stats['scores'])*100:.1f}%)")
    print("="*60)
    
    # Score distribution
    scores = np.array(stats['scores'])
    print("\nScore Distribution:")
    for s in range(0, int(stats['max_score']) + 1):
        count = np.sum(scores == s)
        if count > 0:
            pct = count / len(scores) * 100
            bar = '█' * int(pct / 2)
            print(f"  Score {s:2d}: {count:3d} episodes ({pct:5.1f}%) {bar}")
    print()

def compare_checkpoints(checkpoint_dir="model_checkpoints", num_episodes=50):
    """Compare all available checkpoints."""
    import glob
    
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "policy_ep*.pth")))
    checkpoints.append(os.path.join(checkpoint_dir, "policy_final.pth"))
    
    print(f"Found {len(checkpoints)} checkpoints to compare\n")
    
    results = []
    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            continue
        
        name = os.path.basename(ckpt).replace('.pth', '')
        try:
            stats = evaluate_model(ckpt, num_episodes=num_episodes, render=False)
            results.append((name, stats['avg_score'], stats['max_score']))
            print(f"{name:20s} | Avg: {stats['avg_score']:5.2f} | Max: {stats['max_score']:2d}")
        except Exception as e:
            print(f"{name:20s} | SKIP (incompatible architecture)")
            continue
    
    if results:
        print("\nBest Checkpoint by Average Score:")
        best = max(results, key=lambda x: x[1])
        print(f"  {best[0]} with avg score {best[1]:.2f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained DQN model")
    parser.add_argument("--model", type=str, default="model_checkpoints/policy_final.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to evaluate")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all checkpoints")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_checkpoints(num_episodes=args.episodes)
    else:
        stats = evaluate_model(args.model, num_episodes=args.episodes)
        print_statistics(stats)
