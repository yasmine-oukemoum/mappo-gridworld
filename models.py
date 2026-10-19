
import torch
import torch.nn as nn
from torch.distributions import Categorical



class Actor(nn.Module):
    def __init__(self, obs_dim=4, action_dim=5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs):
        return self.network(obs)

    def get_action(self, obs):
        distribution = Categorical(logits=self.forward(obs))
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

    def evaluate_actions(self, obs, actions):
        distribution = Categorical(logits=self.forward(obs))
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy

class Critic(nn.Module):
    def __init__(self, state_dim=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.network(state)
