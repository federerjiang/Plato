import torch
import torch.multiprocessing as mp

from env import create_atari_env
from model import ActorCritic
from train import train
from test import test
from args import Args


if __name__ == '__main__':
    torch.set_num_threads(1)
    args = Args()
    torch.manual_seed(args.seed)
    env = create_atari_env(args.env)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.share_memory()
    print(model)
    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # p = mp.Process(target=test, args=(args.num_processes, args, model, counter))
    # p.start()
    # processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, counter, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

