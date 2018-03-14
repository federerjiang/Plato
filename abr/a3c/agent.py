import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import numpy as np

import abr.env as env
from abr.load_bw_traces import load_trace
from .a3c import ActorCritic
from .args import Args

S_INFO = 6  # bit_rate, buffer_size, next_seg_size, bandwidth_measurement(throughput and time), seg_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
# VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
VIDEO_BIT_RATE = [10, 15, 20, 25, 30]  # QP according to kvazaar
BUFFER_NORM_FACTOR = 10.0  # used for reward function
SEG_TILL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 50
RAND_RANGE = 1000000


def train(args):
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, _ = load_trace()
    net_env = env.Environment(all_cooked_time, all_cooked_bw)

    model = ActorCritic(state_dim=[S_INFO, S_LEN], action_dim=A_DIM)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    state = np.zeros(S_INFO, S_LEN)
    done = True

    episode_length = 0
    while True:
        # serve video forever
        # the action is from the last decision
        values = []
        log_probs = []
        rewards = []
        entropies = []
        # s_batch = []

        for step in range(args.num_steps):
            episode_length += 1

            inputs = torch.from_numpy(state)
            value, logit = model(inputs)

            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomia().data
            log_prob = log_prob.gather(1, Variable(action))

            # bit_rate = get_bit_rate(prob)
            vp_quality, ad_quality, out_quality = get_bit_rate(prob)

            delay, sleep_time, buffer_size, rebuf, \
            video_seg_size, next_video_seg_sizes, \
            end_of_video, video_seg_remain = net_env.get_video_seg(vp_quality, ad_quality, out_quality)
            tile_map = net_env.get_viewport()

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # # retrieve previous state
            # if len(s_batch) == 0:
            #     state = np.zeros(S_INFO, S_LEN)
            # else:
            #     state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_seg_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_seg_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_seg_remain, SEG_TILL_VIDEO_END_CAP) / float(SEG_TILL_VIDEO_END_CAP)

            # reward is video quality - rebuffer penalty - smooth penalty
            # the reward function is not complete now, needed to be modified later
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            last_bit_rate = bit_rate

            done = end_of_video
            if done:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY

                state = np.zeros(S_INFO, S_LEN)

            # s_batch.append(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            inputs = torch.from_numpy(state)
            value, _ = model(inputs)
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        total_loss = Variable(policy_loss + args.value_loss_coef * value_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()


def get_bit_rate(prob):
    vp_bit_rate = 0
    ad_bit_rate = 0
    out_bit_rate = 0
    return vp_bit_rate, ad_bit_rate, out_bit_rate


if __name__ == "__main__":
    args = Args()
    train(args)
