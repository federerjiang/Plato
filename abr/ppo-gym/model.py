import torch.nn as nn
import torch.nn.functional as F


class ActorModel(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ActorModel, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, a_dim)

    def forward(self, inputs):
        x = F.tanh(self.fc1(inputs))
        x = F.tanh(self.fc2(x))
        logit = self.out(x)
        return logit


class CriticModel(nn.Module):
    def __init__(self, s_dim):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.v = nn.Linear(128, 1)

    def forward(self, inputs):
        x = F.tanh(self.fc1(inputs))
        x = F.tanh(self.fc2(x))
        v = self.v(x)
        return v


