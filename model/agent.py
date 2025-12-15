# model/agent.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# simple MLP policy/target
class DQN(nn.Module):
    def __init__(self, input_dim=17, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        device: torch.device,
        input_dim: int = 17,
        n_actions: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        replay_capacity: int = 100_000,
        seed: int = 42,
        target_update_steps: int = 1000,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.memory = deque(maxlen=replay_capacity)
        self.batch_size = 128
        self.target_update_steps = target_update_steps
        self.steps = 0

        self.policy = DQN(input_dim=input_dim, output_dim=n_actions).to(self.device)
        self.target = DQN(input_dim=input_dim, output_dim=n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, eps: float) -> int:
        # state: numpy array (input_dim,)
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.policy(st)
            return int(torch.argmax(q, dim=1).item())

    def remember(self, s, a, r, ns, done):
        # store raw numpy arrays
        self.memory.append((s, a, r, ns, done))

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        s_batch = np.array([t[0] for t in batch], dtype=np.float32)
        a_batch = np.array([t[1] for t in batch], dtype=np.int64)
        r_batch = np.array([t[2] for t in batch], dtype=np.float32)
        ns_batch = np.array([t[3] for t in batch], dtype=np.float32)
        d_batch = np.array([t[4] for t in batch], dtype=np.float32)  # done as float 0/1

        s_t = torch.from_numpy(s_batch).to(self.device)
        ns_t = torch.from_numpy(ns_batch).to(self.device)
        a_t = torch.from_numpy(a_batch).long().to(self.device).unsqueeze(1)
        r_t = torch.from_numpy(r_batch).float().to(self.device)
        d_t = torch.from_numpy(d_batch).float().to(self.device)

        # current Q
        q_values = self.policy(s_t).gather(1, a_t).squeeze(1)  # shape (B,)

        # Double DQN target
        with torch.no_grad():
            next_actions = torch.argmax(self.policy(ns_t), dim=1, keepdim=True)  # (B,1)
            next_q = self.target(ns_t).gather(1, next_actions).squeeze(1)  # (B,)
            target = r_t + (1.0 - d_t) * (self.gamma * next_q)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.policy.state_dict())
