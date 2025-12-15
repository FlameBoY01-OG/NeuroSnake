# model/dqn.py
import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    MLP: input_dim -> 256 -> 128 -> output_dim
    """
    def __init__(self, input_dim=13, hidden1=256, hidden2=128, output_dim=4):
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
        """
        x: torch tensor (batch x input_dim) or (input_dim,)
        returns numpy arrays (a1, a2)
        """
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            a1 = self.relu1(self.fc1(x))
            a2 = self.relu2(self.fc2(a1))
            return a1.cpu().numpy(), a2.cpu().numpy()
