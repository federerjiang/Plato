# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.conv = nn.Conv1d(state_dim[0], 128, 2)
        self.fc = nn.Linear(128, 128)

        self.actor_linear = nn.Linear(128, action_dim)
        self.critic_linear = nn.Linear(128, 1)

    def forward(self, inputs):
        # all inputs dims should have the same shape
        # each output should also be adjusted to be the same shape
        # consider shape format later
        split_0 = F.relu(self.fc(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv(inputs[:, 2:3, -1]))
        split_3 = F.relu(self.conv(inputs[:, 3:4, -1]))
        split_4 = F.relu(self.conv(inputs[:, 4:5, :self.a_dim]))
        split_5 = F.relu(self.fc(inputs[:, 5:6, -1]))

        merge = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5), 0)
        merge = merge.view(-1, 128)
        return self.critic_linear(merge), self.actor_linear(merge)


