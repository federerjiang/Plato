import gym
import torch
import numpy as np
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.autograd import Variable


# env = gym.make('Amidar-ram-v0')
# env = gym.make('CartPole-v0')
# env = gym.make('PongDeterministic-v4')
# state = env.reset()
# # print(torch.FloatTensor(state))
# # print(state.shape[0])
# buffer_s, buffer_a, buffer_r = [], [], []
# # print(env.action_space.n)
# for _ in range(1000):
#     # env.render()
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#     # print(action)
#     # print(state)
#     print(reward)
#     if done:
#         state = env.reset()
    # print(info)
    # buffer_s.append(state)
    # buffer_r.append(reward)
    # buffer_a.append(action)

# print(buffer_s)
# print(buffer_r)
# print(buffer_a)

# bs = np.vstack(buffer_s)
# print(bs)
# br = np.vstack(buffer_r)
# print(br)
# ba = np.vstack(buffer_a)
# print(ba)
# sar = np.hstack((bs, ba, br))
# print(sar)
#
# s_dim = 3
# a_dim = 1
#
# queue = mp.Queue()
# queue.put(sar)
# sar = queue.get()
# bs = sar[:, :s_dim]
# ba = sar[:, s_dim:s_dim+a_dim]
# br = sar[:, -1:]
# print(bs)
# print(ba)
# print(br)
# rewards = br[::-1]
# print(len(rewards))
# for i in range(len(rewards)):
#     print(rewards[i][0] * 0.9)

# inputs = torch.randn(10)
# print(inputs)
# probs = F.softmax(Variable(inputs))
# print(probs)
# print(probs.multinomial().data)
# print(probs.max())
# _, action = torch.max(probs, 0)
# print(action.data)

import gym
env = gym.make('KungFuMaster-ram-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        # print(env.observation_space.shape[0])
        action = env.action_space.sample()
        print('action', action)
        observation, reward, done, info = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
