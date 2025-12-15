# model/replay_memory.py
import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

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
