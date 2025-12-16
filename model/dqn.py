# model/dqn.py
import torch
import torch.nn as nn

class SimpleDQN(nn.Module):
    """
    Simple 3-layer DQN that actually works for Snake.
    Proven architecture: input -> 128 -> 128 -> 4
    """
    def __init__(self, input_dim=11, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_activations(self, x):
        """Extract activations for visualization."""
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # Layer by layer
            h1 = self.net[0](x)  # First linear
            h1_act = self.net[1](h1)  # ReLU
            h2 = self.net[2](h1_act)  # Second linear
            h2_act = self.net[3](h2)  # ReLU
            output = self.net[4](h2_act)  # Output layer
            
            return (
                h1_act.cpu().numpy(),
                h2_act.cpu().numpy(),
                output.cpu().numpy(),
                {},  # Empty weights dict for compatibility
                {},
                output.cpu().numpy()
            )

# Alias for backward compatibility
DuelingDQN = SimpleDQN


# Legacy DQN for backward compatibility
class DQN(nn.Module):
    """
    Standard MLP DQN - kept for compatibility.
    Use DuelingDQN for better performance.
    """
    def __init__(self, input_dim=32, hidden1=256, hidden2=128, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        q = self.out(x)
        return q

    def get_activations(self, x):
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            a1 = self.relu1(self.fc1(x))
            a2 = self.relu2(self.fc2(a1))
            return a1.cpu().numpy(), a2.cpu().numpy()
