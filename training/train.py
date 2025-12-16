# training/train.py
import os
import time
import numpy as np
import torch
from env.snake_env import SnakeEnv
from model.agent import DQNAgent

def train(
    num_episodes=1000,
    max_steps_per_episode=1000,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay_steps=80000,
    save_every=100,
    model_dir="model_checkpoints",
    device=None,
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Device:", device)
    print("\n=== Training Simple DQN Agent ===")

    env = SnakeEnv(width=20, height=20)
    state_dim = env.reset().shape[0]
    print(f"State dimension: {state_dim}")
    print(f"Training for {num_episodes} episodes\n")
    
    agent = DQNAgent(
        input_dim=state_dim,
        output_dim=4,
        device=device,
        gamma=gamma,
        lr=1e-3,
        batch_size=64,
        memory_capacity=50000,
        use_prioritized_replay=False,
        target_update_steps=1000
    )

    os.makedirs(model_dir, exist_ok=True)

    total_steps = 0
    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        state = env.reset(randomize_start=True)
        ep_reward = 0.0
        ep_loss = []

        for step in range(max_steps_per_episode):
            eps = eps_end + (eps_start - eps_end) * max(0, (eps_decay_steps - total_steps)) / eps_decay_steps
            
            action = agent.select_action(state, eps)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) > 500:
                loss = agent.optimize(gradient_steps=1)
                if loss is not None:
                    ep_loss.append(loss)

            state = next_state
            ep_reward += reward
            total_steps += 1

            if done:
                break

        episode_rewards.append(ep_reward)

        if ep % 10 == 0:
            avg_recent = float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 1 else ep_reward
            avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
            print(f"Ep: {ep:4d} | Reward: {ep_reward:6.2f} | Avg50: {avg_recent:6.2f} | Score: {env.score:2d} | Steps: {total_steps} | Eps: {eps:.3f} | Loss: {avg_loss:.4f}")

        if ep % save_every == 0:
            fn = os.path.join(model_dir, f"policy_ep{ep}.pth")
            agent.save(fn)
            print(f"Saved: {fn}")

    agent.save(os.path.join(model_dir, "policy_final.pth"))
    print(f"\nTraining complete! Final avg reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Model saved: {os.path.join(model_dir, 'policy_final.pth')}")

if __name__ == "__main__":
    train()
