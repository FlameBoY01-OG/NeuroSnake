# model/agent.py
import random
import os
from collections import deque, namedtuple
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dqn import SimpleDQN as DuelingDQN
from model.replay_memory import PrioritizedReplayMemory

Transition = namedtuple("Transition", ("s", "a", "r", "s2", "d"))

class DQNAgent:
    """
    DQN Agent with Double DQN and target network.
    
    Features:
    - Double DQN for reduced overestimation bias
    - Target network with periodic hard updates
    - Experience replay buffer
    - Gradient clipping for training stability
    - Optional prioritized replay (configurable)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 4,
        device: torch.device = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        memory_capacity: int = 200_000,
        target_update_steps: int = 1000,
        use_prioritized_replay: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 100_000,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.target_update_steps = int(target_update_steps)
        
        # Prioritized replay parameters
        self.use_prioritized_replay = use_prioritized_replay
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_steps = per_beta_steps

        # Use Dueling DQN architecture
        self.policy = DuelingDQN(input_dim=input_dim, output_dim=output_dim).to(self.device)
        self.target = DuelingDQN(input_dim=input_dim, output_dim=output_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()  # Target network always in eval mode
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Use prioritized replay memory
        if use_prioritized_replay:
            self.memory = PrioritizedReplayMemory(capacity=memory_capacity, alpha=per_alpha)
        else:
            from model.replay_memory import ReplayMemory
            self.memory = ReplayMemory(capacity=memory_capacity)
        
        self.steps_done = 0
        self.rng = np.random.default_rng()

    def select_action(self, state: np.ndarray, eps: float = 0.05) -> int:
        """
        state: 1D numpy array
        returns int action
        """
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.output_dim))
        # greedy
        st = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy(st)
        return int(torch.argmax(q[0]).item())

    def remember(self, s, a, r, s2, d):
        """Store transition in replay memory."""
        self.memory.push(np.array(s, dtype=np.float32), int(a), float(r), np.array(s2, dtype=np.float32), bool(d))

    def optimize(self, gradient_steps: int = 1):
        """
        Perform gradient descent steps with prioritized replay.
        Returns average loss over gradient steps.
        """
        if len(self.memory) < self.batch_size:
            return None
        
        total_loss = 0.0
        for _ in range(gradient_steps):
            # Compute beta for importance sampling (anneals to 1.0)
            if self.use_prioritized_replay:
                beta = self.per_beta_start + (self.per_beta_end - self.per_beta_start) * \
                       min(1.0, self.steps_done / self.per_beta_steps)
                sample_result = self.memory.sample(self.batch_size, beta)
                if sample_result is None:
                    return None
                s_batch_np, a_batch_np, r_batch_np, s2_batch_np, d_batch_np, indices, weights = sample_result
                
                # Convert to tensors
                s_batch = torch.from_numpy(s_batch_np).float().to(self.device)
                a_batch = torch.from_numpy(a_batch_np).to(self.device)
                r_batch = torch.from_numpy(r_batch_np).to(self.device)
                s2_batch = torch.from_numpy(s2_batch_np).float().to(self.device)
                d_batch = torch.from_numpy(d_batch_np).to(self.device)
                weights_tensor = torch.from_numpy(weights).float().to(self.device)
            else:
                s_batch_np, a_batch_np, r_batch_np, s2_batch_np, d_batch_np = self.memory.sample(self.batch_size)
                s_batch = torch.from_numpy(s_batch_np).float().to(self.device)
                a_batch = torch.from_numpy(a_batch_np).to(self.device)
                r_batch = torch.from_numpy(r_batch_np).to(self.device)
                s2_batch = torch.from_numpy(s2_batch_np).float().to(self.device)
                d_batch = torch.from_numpy(d_batch_np).to(self.device)
                weights_tensor = torch.ones(self.batch_size, device=self.device)

            # Q(s,a) - current Q values
            q_values = self.policy(s_batch)
            q_sa = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)

            # Double DQN: use policy network to select action, target network to evaluate
            with torch.no_grad():
                # Policy network selects best action
                q_next_policy = self.policy(s2_batch)
                best_actions = q_next_policy.argmax(dim=1)
                
                # Target network evaluates that action
                q_next_target = self.target(s2_batch)
                q_next_max = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                
                not_done = (1 - d_batch.float())
                target = r_batch + self.gamma * q_next_max * not_done

            # Compute TD-errors for priority updates
            td_errors = (q_sa - target).detach()
            
            # Weighted MSE loss (importance sampling)
            loss = (weights_tensor * (q_sa - target).pow(2)).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.optimizer.step()

            # Update priorities in replay buffer
            if self.use_prioritized_replay:
                self.memory.update_priorities(indices, td_errors.cpu().numpy())

            total_loss += loss.item()
            self.steps_done += 1
            
            # Soft update target network
            if self.steps_done % self.target_update_steps == 0:
                self.target.load_state_dict(self.policy.state_dict())
        
        return total_loss / gradient_steps

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.target.load_state_dict(data.get("target", data["policy"]))
        if "optimizer" in data:
            try:
                self.optimizer.load_state_dict(data["optimizer"])
                # move optimizer tensors to device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            except Exception:
                pass

    def get_activations(self, state: np.ndarray):
        """
        Enhanced activation extraction for Dueling DQN visualization.
        Returns comprehensive activation data for explainable AI rendering.
        
        Returns:
        - activations: dict with keys for each network component
        - weights: dict with weight matrices for visualization
        """
        st = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract all intermediate activations from Dueling DQN
            h1, h2, output, _, _, _ = self.policy.get_activations(st)
            
            # Convert to simple list format
            activations = [state.copy(), h1[0], h2[0], output[0]]
            
            # Extract weights (input->h1, h1->h2, h2->output)
            weights = [
                self.policy.net[0].weight.detach().cpu().numpy(),  # input -> h1
                self.policy.net[2].weight.detach().cpu().numpy(),  # h1 -> h2
                self.policy.net[4].weight.detach().cpu().numpy(),  # h2 -> output
            ]
            
            return activations, weights
