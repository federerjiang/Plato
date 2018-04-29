from ewma import EwmaBandwidthEstimator
import time
from env_hd_partial import Environment
from args import Args, LSTMPredict


class PART:
    def __init__(self):
        self.estimator = EwmaBandwidthEstimator(8, 3, 10)
        self.default_action = (5, 5)

    def action(self, bandwidth, delay, vp_sizes, ad_sizes, out_sizes):
        self.estimator.sample(delay, bandwidth)
        bandwidth_prediction = self.estimator.get_estimate()
        vp_q = ad_q = out_q = 5
        budget = bandwidth_prediction - vp_sizes[vp_q] - ad_sizes[ad_q] - out_sizes[out_q]
        while True:
            if budget <= 0:
                break
            else:
                for q in range(0, 5):
                    if vp_sizes[q] <= budget:
                        budget = budget - vp_sizes[q]
                        vp_q = q
                        break
                for q in range(0, 5):
                    if ad_sizes[q] <= budget:
                        budget = budget - ad_sizes[q]
                        ad_q = q
                        break
                for q in range(0, 5):
                    if out_sizes[q] <= budget:
                        budget = budget - out_sizes[q]
                        out_q = q
                        break
                break
        # print('bandwidth: ', bandwidth)
        # print('delay: ', delay)
        # print('vp: ', vp_sizes)
        # print('ad: ', ad_sizes)
        # print('out: ', out_sizes)
        return vp_q, ad_q, out_q


def test(rank, args,
         all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit):

    env = Environment(args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit, random_seed=args.seed + rank)
    model = PART()
    action = model.default_action
    bandwidth, delay, vp_sizes, ad_sizes, out_sizes, done, (rebuf, cv, blank_ratio, reward, real_vp_bitrate, smooth) = \
        env.step(action)
    state_time = time.time()
    episode_length = 0
    # log = open('new-result-1/test-vp-log20000.txt', 'w')
    # log = open('results-3/log20000.txt', 'w')
    # log = open('train_norway_result-2/test_log3000.txt', 'w')
    log = open('result-1/log-hd-partial.txt', 'w')
    while True:
        episode_length += 1
        action = model.action(bandwidth, delay, vp_sizes, ad_sizes, out_sizes)
        bandwidth, delay, vp_sizes, ad_sizes, out_sizes, done, (rebuf, cv, blank_ratio, reward, real_vp_bitrate, smooth) \
            = env.step(action)
        update = True
        if update:
            print("Time {}, action ({},{},{}), bitrate {:.3f}, rebuf {:.3f}, cv {:.3f}, smooth {:.3f}, reward {:.3f}, episode {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - state_time)),
                action[0], action[1], action[2], real_vp_bitrate, rebuf, cv, smooth,
                reward, episode_length))
            log.write('action: ' + str(1) + ' (' + str(action[0]) + ',' + str(action[1]) + ',' + str(action[2])
                      + ') rebuf: ' + str(rebuf) + ' cv: ' + str(cv) + ' black_ratio: ' + str(blank_ratio) +
                      ' smooth: ' + str(smooth) + ' bitrate: ' + str(real_vp_bitrate) + ' reward: ' + str(reward)
                      + ' episode: ' + str(episode_length) + '\n')
            # print('Time {}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - state_time))))
            # print('time: ', time.gmtime(time.time() - state_time))
            # time.sleep(0.5)
            pass
        if done:
            env.reset()
            action = model.default_action
            bandwidth, delay, vp_sizes, ad_sizes, out_sizes, done, (rebuf, cv, blank_ratio, reward, real_vp_bitrate, smooth) = \
                env.step(action)
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
    all_cooked_time, all_cooked_bw, _ = load_trace(bw_trace_folder)
    all_vp_time, all_vp_unit = load_viewport_unit(vp_trace_folder)
    test(1, args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit)




