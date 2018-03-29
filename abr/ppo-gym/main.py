import gym
import os

import torch
import torch.optim as optim
import torch.multiprocessing as mp

from worker import worker
from chief import chief
from model import ActorModel, CriticModel


class Args:
    def __init__(self):
        self.batch_size = 16
        self.a_lr = 3e-4
        self.c_lr = 3e-3
        self.gamma = 0.99
        self.gae = 0.95
        self.clip = 0.2
        self.ent_coef = 0.
        self.num_epoch = 10
        self.num_steps = 10000
        self.min_batch_size = 640
        self.num_processes = 1
        self.update_steps = 100
        self.update_threshold = self.num_processes - 1
        self.max_episode_length = 100
        self.seed = 30
        # self.env_name = 'Amidar-ram-v0'  # https://gym.openai.com/envs/Amidar-ram-v0/
        # self.env_name = 'Pendulum-v0'
        # self.env_name = 'CartPole-v0'
        self.env_name = 'KungFuMaster-ram-v0'
        self.s_dim = 128
        self.a_dim = 14


class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self, val=True):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    args = Args()
    torch.manual_seed(args.seed)
    env = gym.make(args.env_name)
    # s_dim = env.observation_space.shape[0]
    # a_dim = env.action_space.shape[0]
    s_dim = args.s_dim
    a_dim = args.a_dim

    actor_model_shared = ActorModel(s_dim, a_dim)
    critic_model_shared = CriticModel(s_dim)
    actor_model_shared.share_memory()
    critic_model_shared.share_memory()

    actor_optimizer = optim.Adam(actor_model_shared.parameters(), lr=args.a_lr)
    critic_optimizer = optim.Adam(critic_model_shared.parameters(), lr=args.c_lr)

    update_event, rolling_event = mp.Event(), mp.Event()
    update_event.clear()  # not update now
    rolling_event.set()  # start to roll out
    queue = mp.Queue()  # workers put data in this queue
    counter = Counter()
    queue_size = Counter()

    processes = []
    p = mp.Process(target=chief, args=(args, actor_model_shared, critic_model_shared,
                                       update_event, rolling_event, queue, counter, queue_size,
                                       actor_optimizer, critic_optimizer))
    p.start()
    processes.append(p)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=worker, args=(args, actor_model_shared, critic_model_shared,
                                            update_event, rolling_event, queue, counter, queue_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()




