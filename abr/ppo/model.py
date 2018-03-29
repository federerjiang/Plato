# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorModel(nn.Module):
    def __init__(self, a_dim=180):
        super(ActorModel, self).__init__()
        self.a_dim = a_dim
        self.fc0 = nn.Linear(1, 128)
        self.fc1 = nn.Linear(1, 128)
        self.conv2 = nn.Conv1d(1, 128, 3)
        self.conv3 = nn.Conv1d(1, 128, 3)
        self.conv4 = nn.Conv1d(1, 128, 3)
        self.conv5 = nn.Conv1d(1, 128, 3)
        self.conv6 = nn.Conv1d(1, 128, 3)
        self.fc7 = nn.Linear(1, 128)
        self.fc8 = nn.Linear(1, 128)
        self.fc9 = nn.Linear(1, 128)
        self.fc10 = nn.Linear(1, 128)
        self.fc = nn.Linear(21*128, 128)
        self.actor_linear = nn.Linear(128, self.a_dim)

    def forward(self, inputs, batch_size=1):
        split_0 = F.relu(self.fc0(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc1(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv2(inputs[:, 2:3, :5])).view(batch_size, -1)
        split_3 = F.relu(self.conv3(inputs[:, 3:4, :5])).view(batch_size, -1)
        split_4 = F.relu(self.conv4(inputs[:, 4:5, :5])).view(batch_size, -1)
        split_5 = F.relu(self.conv5(inputs[:, 5:6, :5])).view(batch_size, -1)
        split_6 = F.relu(self.conv6(inputs[:, 6:7, :5])).view(batch_size, -1)
        split_7 = F.relu(self.fc7(inputs[:, 7:8, -1]))
        split_8 = F.relu(self.fc8(inputs[:, 8:9, -1]))
        split_9 = F.relu(self.fc9(inputs[:, 9:10, -1]))
        split_10 = F.relu(self.fc10(inputs[:, 10:11, -1]))

        # print(split_0.shape, split_2.shape)
        merge = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5,
                           split_6, split_7, split_8, split_9, split_10), 1)
        merge = merge.view(batch_size, -1)
        # print(merge.shape)
        fc_out = F.relu(self.fc(merge))
        logit = self.actor_linear(fc_out)
        return logit


class CriticModel(nn.Module):
    def __init__(self, a_dim=1):
        super(CriticModel, self).__init__()
        self.a_dim = a_dim
        self.fc0 = nn.Linear(1, 128)
        self.fc1 = nn.Linear(1, 128)
        self.conv2 = nn.Conv1d(1, 128, 3)
        self.conv3 = nn.Conv1d(1, 128, 3)
        self.conv4 = nn.Conv1d(1, 128, 3)
        self.conv5 = nn.Conv1d(1, 128, 3)
        self.conv6 = nn.Conv1d(1, 128, 3)
        self.fc7 = nn.Linear(1, 128)
        self.fc8 = nn.Linear(1, 128)
        self.fc9 = nn.Linear(1, 128)
        self.fc10 = nn.Linear(1, 128)
        self.fc = nn.Linear(21*128, 128)
        self.critic_linear = nn.Linear(128, self.a_dim)

    def forward(self, inputs, batch_size=1):
        split_0 = F.relu(self.fc0(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc1(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv2(inputs[:, 2:3, :5])).view(batch_size, -1)
        split_3 = F.relu(self.conv3(inputs[:, 3:4, :5])).view(batch_size, -1)
        split_4 = F.relu(self.conv4(inputs[:, 4:5, :5])).view(batch_size, -1)
        split_5 = F.relu(self.conv5(inputs[:, 5:6, :5])).view(batch_size, -1)
        split_6 = F.relu(self.conv6(inputs[:, 6:7, :5])).view(batch_size, -1)
        split_7 = F.relu(self.fc7(inputs[:, 7:8, -1]))
        split_8 = F.relu(self.fc8(inputs[:, 8:9, -1]))
        split_9 = F.relu(self.fc9(inputs[:, 9:10, -1]))
        split_10 = F.relu(self.fc10(inputs[:, 10:11, -1]))

        # print(split_0.shape, split_2.shape)
        merge = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5,
                           split_6, split_7, split_8, split_9, split_10), 1)
        merge = merge.view(batch_size, -1)
        # print(merge.shape)
        fc_out = F.relu(self.fc(merge))
        v = self.critic_linear(fc_out)
        return v
