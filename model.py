import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, seed):
        """Initialize parameters and build model.
        Params
        ======
            in_actor (int): Dimension of each state
            out_actor (int): Dimension of each action
            hidden_in_actor (int): Number of nodes in first hidden layer
            hidden_out_actor (int): Number of nodes in second hidden layer
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_actor, hidden_in_actor)
        self.fc2 = nn.Linear(hidden_in_actor, hidden_out_actor)
        self.fc3 = nn.Linear(hidden_out_actor, out_actor)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, in_critic, hidden_in_critic, hidden_out_critic, seed):
        """Initialize parameters and build model.
        Params
        ======
            in_critic (int): Dimension of each state
            hidden_in_critic (int): Number of nodes in the first hidden layer
            hidden_out_critic (int): Number of nodes in the second hidden layer
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_critic, hidden_in_critic)
        self.fc2 = nn.Linear(hidden_in_critic, hidden_out_critic)
        self.fc3 = nn.Linear(hidden_out_critic, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xin = torch.cat((state,action), dim=1)
        x = F.relu(self.fc1(xin))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


