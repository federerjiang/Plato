# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, a_dim=180):
        super(ActorCritic, self).__init__()
        self.a_dim = a_dim
        # actor model
        self.a_fc0 = nn.Linear(1, 128)
        self.a_fc1 = nn.Linear(1, 128)
        self.a_conv2 = nn.Conv1d(1, 128, 3)
        self.a_conv3 = nn.Conv1d(1, 128, 3)
        self.a_conv4 = nn.Conv1d(1, 128, 3)
        self.a_conv5 = nn.Conv1d(1, 128, 3)
        self.a_conv6 = nn.Conv1d(1, 128, 3)
        self.a_fc7 = nn.Linear(1, 128)
        self.a_fc8 = nn.Linear(1, 128)
        self.a_fc9 = nn.Linear(1, 128)
        self.a_fc10 = nn.Linear(1, 128)
        self.a_fc = nn.Linear(21*128, 128)
        self.a_actor_linear = nn.Softmax(nn.Linear(128, self.a_dim))

        # critic model
        self.c_fc0 = nn.Linear(1, 128)
        self.c_fc1 = nn.Linear(1, 128)
        self.c_conv2 = nn.Conv1d(1, 128, 3)
        self.c_conv3 = nn.Conv1d(1, 128, 3)
        self.c_conv4 = nn.Conv1d(1, 128, 3)
        self.c_conv5 = nn.Conv1d(1, 128, 3)
        self.c_conv6 = nn.Conv1d(1, 128, 3)
        self.c_fc7 = nn.Linear(1, 128)
        self.c_fc8 = nn.Linear(1, 128)
        self.c_fc9 = nn.Linear(1, 128)
        self.c_fc10 = nn.Linear(1, 128)
        self.c_fc = nn.Linear(21 * 128, 128)
        self.c_critic_linear = nn.Linear(128, 1)

    def forward(self, inputs, batch_size=1):
        # actor
        split_0 = F.relu(self.a_fc0(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.a_fc1(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.a_conv2(inputs[:, 2:3, 3:8])).view(batch_size, -1)
        split_3 = F.relu(self.a_conv3(inputs[:, 3:4, 3:8])).view(batch_size, -1)
        split_4 = F.relu(self.a_conv4(inputs[:, 4:5, :5])).view(batch_size, -1)
        split_5 = F.relu(self.a_conv5(inputs[:, 5:6, :5])).view(batch_size, -1)
        split_6 = F.relu(self.a_conv6(inputs[:, 6:7, :5])).view(batch_size, -1)
        split_7 = F.relu(self.a_fc7(inputs[:, 7:8, -1]))
        split_8 = F.relu(self.a_fc8(inputs[:, 8:9, -1]))
        split_9 = F.relu(self.a_fc9(inputs[:, 9:10, -1]))
        split_10 = F.relu(self.a_fc10(inputs[:, 10:11, -1]))

        # print(split_0.shape, split_2.shape)
        merge = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5,
                           split_6, split_7, split_8, split_9, split_10), 1)
        merge = merge.view(batch_size, -1)
        # print(merge.shape)
        fc_out = F.relu(self.a_fc(merge))
        logit = self.a_actor_linear(fc_out)

        # critic
        split_0 = F.relu(self.c_fc0(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.c_fc1(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.c_conv2(inputs[:, 2:3, 3:8])).view(batch_size, -1)
        split_3 = F.relu(self.c_conv3(inputs[:, 3:4, 3:8])).view(batch_size, -1)
        split_4 = F.relu(self.c_conv4(inputs[:, 4:5, :5])).view(batch_size, -1)
        split_5 = F.relu(self.c_conv5(inputs[:, 5:6, :5])).view(batch_size, -1)
        split_6 = F.relu(self.c_conv6(inputs[:, 6:7, :5])).view(batch_size, -1)
        split_7 = F.relu(self.c_fc7(inputs[:, 7:8, -1]))
        split_8 = F.relu(self.c_fc8(inputs[:, 8:9, -1]))
        split_9 = F.relu(self.c_fc9(inputs[:, 9:10, -1]))
        split_10 = F.relu(self.c_fc10(inputs[:, 10:11, -1]))

        # print(split_0.shape, split_2.shape)
        merge = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5,
                           split_6, split_7, split_8, split_9, split_10), 1)
        merge = merge.view(batch_size, -1)
        # print(merge.shape)
        fc_out = F.relu(self.c_fc(merge))
        v = self.c_critic_linear(fc_out)

        return logit, v