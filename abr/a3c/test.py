import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from env import Environment
from model import ActorCritic


def test(rank, args, shared_model, counter,
         all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit):
    torch.manual_seed(args.seed + rank)

    env = Environment(args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit, random_seed=args.seed + rank)

    model = ActorCritic()

    model.eval()

    state = env.reset()
    reward_sum = 0
    done = True

    state_time = time.time()

    actions = deque(maxlen=100)
    episode_length = 0
    log = open('log.txt', 'r+')
    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())

        state = Variable(torch.FloatTensor(state))
        logit, value = model(state.view(-1, 11, 8))
        prob = F.softmax(logit, dim=1)
        _, action = torch.max(prob, 1)
        state, reward, done, (action, vp_quality, ad_quality, out_quality, rebuf, cv, blank_ratio, reward) \
            = env.step(action.data.numpy()[0])
        done = done or episode_length >= args.max_episode_length
        done = True
        reward_sum = reward

        # actions.append(action[0, 0])
        # if actions.count(actions[0]) >= actions.maxlen:
        #     done = True

        if done:
            print("Time {}, action {}, ({},{},{}), rebuf {:.3f}, cv {:.3f}, black_ratio {:.3f}, reward {:.3f}, episode {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - state_time)),
                action, vp_quality, ad_quality, out_quality, rebuf, cv, blank_ratio,
                reward_sum, episode_length))
            log.write('action: ' + str(action) + ' (' + str(vp_quality) + ',' + str(ad_quality) + ',' + str(out_quality)
                      + ') rebuf: ' + str(rebuf) + ' black_ratio: ' + str(blank_ratio) + ' reward: ' + str(reward)
                      + ' episode: ' + str(episode_length) + '\n')
            # print('Time {}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - state_time))))
            # print('time: ', time.gmtime(time.time() - state_time))
            if episode_length % 100 == 0:
                path = 'results/actor.pt-' + str(episode_length / 100)
                torch.save(model.state_dict(), path)
                print('saved one model in epoch:', episode_length)

            # episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(1)