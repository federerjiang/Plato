import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from env import Environment
from model import ActorCritic
from args import Args, LSTMPredict


def test(rank, args, model_path,
         all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit):
    torch.manual_seed(args.seed + rank)

    env = Environment(args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit, random_seed=args.seed + rank)

    model = ActorCritic()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    state = env.reset()
    state_time = time.time()
    episode_length = 0
    # log = open('log-2.txt', 'w')
    while True:
        episode_length += 1
        state = Variable(torch.FloatTensor(state))
        # print('state', state)
        logit, value = model(state.view(-1, 11, 8))
        prob = F.softmax(logit, dim=1)
        _, action = torch.max(prob, 1)
        state, reward, done, (action, vp_quality, ad_quality, out_quality, rebuf, cv, blank_ratio, reward) \
            = env.step(action.data.numpy()[0])
        update = True

        if update:
            print("Time {}, action {}, ({},{},{}), rebuf {:.3f}, cv {:.3f}, black_ratio {:.3f}, reward {:.3f}, episode {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - state_time)),
                action, vp_quality, ad_quality, out_quality, rebuf, cv, blank_ratio,
                reward, episode_length))
            # log.write('action: ' + str(action) + ' (' + str(vp_quality) + ',' + str(ad_quality) + ',' + str(out_quality)
            #           + ') rebuf: ' + str(rebuf) + ' black_ratio: ' + str(blank_ratio) + ' reward: ' + str(reward)
            #           + ' episode: ' + str(episode_length) + '\n')
            # print('Time {}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - state_time))))
            # print('time: ', time.gmtime(time.time() - state_time))
            time.sleep(0.5)
        if done:
            state = env.reset()


if __name__ == '__main__':
    import sys, inspect, os

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
    model_path = 'results/actor.pt-6000'
    test(1, args, model_path, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit)
