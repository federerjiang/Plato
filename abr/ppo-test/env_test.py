import gym

envs = ['SeaquestNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'QberNoFrameskip-v4',
        'HalfCheetah-v1',
        'Hopper-v1',
        'PongDeterministic-v4',
        ]
env = gym.make(envs[5])
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
print('s_dim: ', s_dim)
print('a_dim: ', a_dim)
# env.reset()
# env.render()
#
# env.monitor.start('/tmp/reacher-1', force=True)
for i_episode in range(101):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print('reward', reward)

        if done:
            print('Episode finished after {} timesteps'.format(t+1))
            break
