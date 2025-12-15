# training/train.py
import os
import torch
from env.snake_env import SnakeEnv
from model.agent import DQNAgent
import numpy as np

def train_stage(agent, grid_size, episodes, max_steps=5000):
    env = SnakeEnv(width=grid_size, height=grid_size)
    total_steps = 0
    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            # linear epsilon decay fast so we reach exploitation quickly
            eps = max(0.01, 1.0 - total_steps / 5000.0)
            a = agent.select_action(s, eps)
            ns, r, done = env.step(a)
            agent.remember(s, a, r, ns, done)
            loss = agent.optimize()
            s = ns
            ep_reward += r
            total_steps += 1
            steps += 1

        if ep % 50 == 0 or ep == 1:
            print(f"[{grid_size}x{grid_size}] Ep {ep:4d} | Reward {ep_reward:7.2f} | StepsEp {steps:4d} | TotalSteps {total_steps:6d} | Eps {eps:.3f}")

if __name__ == "__main__":
    # CUDA-aware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device name:", torch.cuda.get_device_name(device))
        torch.backends.cudnn.benchmark = True

    agent = DQNAgent(device=device)

    # curriculum: 10x10 -> 15x15 -> 20x20
    train_stage(agent, grid_size=10, episodes=800)
    train_stage(agent, grid_size=15, episodes=800)
    train_stage(agent, grid_size=20, episodes=1200)

    os.makedirs("model_checkpoints", exist_ok=True)
    agent.save("model_checkpoints/policy_final.pth")
    print("Training complete — model saved to model_checkpoints/policy_final.pth")
