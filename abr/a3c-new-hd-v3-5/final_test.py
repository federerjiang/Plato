import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from fixed_env import Environment
from model import ActorCritic
from args import Args, LSTMPredict


def test(rank, args, model_path,
         all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit, num):
    torch.manual_seed(args.seed + rank)

    env = Environment(args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit, random_seed=args.seed + rank)

    model = ActorCritic()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    state = env.reset()
    state_time = time.time()
    episode_length = 0
    # log = open('new-result-1/test-vp-log20000.txt', 'w')
    # log = open('results-3/log20000.txt', 'w')
    # log = open('train_norway_result-2/test_log3000.txt', 'w')
    log = open('result-1/log-' + str(num) + '.txt', 'w')
    while True:
        episode_length += 1
        state = Variable(torch.FloatTensor(state))
        # print('state', state)
        logit, value = model(state.view(-1, 11, 8))
        prob = F.softmax(logit, dim=1)
        _, action = torch.max(prob, 1)
        state, reward, done, (action, vp_quality, ad_quality, out_quality, rebuf, cv, blank_ratio, reward, real_vp_bitrate, smooth) \
            = env.step(action.data.numpy()[0])
        update = True

        if update:
            print("Time {}, action {}, ({},{},{}), bitrate {:.3f}, rebuf {:.3f}, cv {:.3f}, smooth {:.3f}, reward {:.3f}, episode {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - state_time)),
                action, vp_quality, ad_quality, out_quality, real_vp_bitrate, rebuf, cv, smooth,
                reward, episode_length))
            log.write('action: ' + str(action) + ' (' + str(vp_quality) + ',' + str(ad_quality) + ',' + str(out_quality)
                      + ') rebuf: ' + str(rebuf) + ' cv: ' + str(cv) + ' bitrate: ' + str(real_vp_bitrate) + ' smooth: ' + str(smooth) + ' reward: ' + str(reward)
                      + ' episode: ' + str(episode_length) + '\n')
            # log.write(str())
            # print('Time {}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - state_time))))
            # print('time: ', time.gmtime(time.time() - state_time))
            # time.sleep(0.5)
        if done:
            state = env.reset()
        if episode_length == 50000:
            log.close()
            break


if __name__ == '__main__':
    import sys, inspect, os

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from load_bw_traces import load_trace
    from load_viewport_trace import load_viewport_unit

    # bw_trace_folder = '../../datasets/bw_trace/test_sim_belgium/'
    # bw_trace_folder = '../../datasets/bw_trace/train_sim_traces/'
    bw_trace_folder = '../../datasets/bw_trace/sim_belgium/'
    vp_trace_folder = '../../datasets/viewport_trace/RL_new_cooked_test_dataset/'
    args = Args()
    torch.manual_seed(args.seed)
    all_cooked_time, all_cooked_bw, _ = load_trace(bw_trace_folder)
    all_vp_time, all_vp_unit = load_viewport_unit(vp_trace_folder)
    # nums = [20000, 40000, 45000, 55000]
    nums = [170000]
    for num in nums:
        model_path = 'result-1/actor.pt-' + str(num)  # 20000, 40000, 45000, 55000
        test(1, args, model_path, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit, num)
