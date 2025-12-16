# model/replay_memory.py
import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay with Sum Tree.
    
    Samples important transitions (high TD-error) more frequently.
    Critical for learning from rare success events (eating food, avoiding death).
    
    Key parameters:
    - alpha: how much prioritization (0 = uniform, 1 = fully prioritized)
    - beta: importance sampling weight (starts low, anneals to 1)
    - epsilon: small constant to ensure non-zero priority
    """
    def __init__(self, capacity=200_000, alpha=0.6, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.epsilon = epsilon
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """Add transition with maximum priority (will be refined after training)."""
        self.memory.append(Transition(state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size, beta=0.4):
        """
        Sample batch with prioritization.
        Returns: (states, actions, rewards, next_states, dones, indices, weights)
        
        indices: for updating priorities after computing TD-error
        weights: importance sampling weights to correct bias
        """
        if len(self.memory) < batch_size:
            return None
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        priorities = priorities ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices according to priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize to [0, 1]
        
        # Extract transitions
        batch = [self.memory[idx] for idx in indices]
        states = np.vstack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.vstack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.uint8)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-errors from training."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)


# Standard replay memory for backward compatibility
class ReplayMemory:
    def __init__(self, capacity=200_000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states = np.vstack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.vstack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.uint8)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
