# import gym
import os

import torch
import torch.optim as optim
import multiprocessing as mp

from worker import worker
from chief import chief
from model import ActorModel, CriticModel
from args import Args, LSTMPredict
# from abr.load_bw_traces import load_trace
# from abr.load_viewport_trace import load_viewport_unit


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

    import sys, inspect
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from load_bw_traces import load_trace
    from load_viewport_trace import load_viewport_unit
    bw_trace_folder = '../../datasets/bw_trace/sim_belgium/'
    vp_trace_folder = '../../datasets/viewport_trace/new_cooked_train_dataset/'
    args = Args()
    torch.manual_seed(args.seed)
    all_cooked_time, all_cooked_bw, _ = load_trace(bw_trace_folder)
    all_vp_time, all_vp_unit = load_viewport_unit(vp_trace_folder)

    actor_model_shared = ActorModel()
    critic_model_shared = CriticModel()
    actor_model_shared.share_memory()
    critic_model_shared.share_memory()

    # actor_optimizer = optim.Adam(actor_model_shared.parameters(), lr=args.a_lr)
    # critic_optimizer = optim.Adam(critic_model_shared.parameters(), lr=args.c_lr)

    update_events = []
    rolling_events = []
    for rank in range(args.num_processes):
        update_events.append(mp.Event())
        update_events[rank].clear()
        rolling_events.append(mp.Event())
        rolling_events[rank].set()
    # update_event, rolling_event = mp.Event(), mp.Event()
    # update_event.clear()  # not update now
    # rolling_event.set()  # start to roll out
    state_queue = mp.Queue()  # workers put data in this queue
    queue = mp.Queue()
    counter = Counter()
    queue_size = Counter()

    processes = []
    p = mp.Process(target=chief, args=(args, actor_model_shared, critic_model_shared,
                                       update_events, rolling_events, state_queue, queue, counter, queue_size,
                                       ))
    p.start()
    processes.append(p)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=worker, args=(rank, args, actor_model_shared, critic_model_shared,
                                            update_events, rolling_events, state_queue, queue, counter, queue_size,
                                            all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()




