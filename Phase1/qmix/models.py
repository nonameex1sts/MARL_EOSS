import torch
import torch.nn as nn

class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        return self.net(obs)

class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
            nn.Softmax(dim=1)  # Softmax to get probabilities for agent selection
        )

    def forward(self, q_values, global_state):
        x = torch.cat([q_values.view(1, -1), global_state.view(1, -1)], dim=1)
        return self.net(x)
