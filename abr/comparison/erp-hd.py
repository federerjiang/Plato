from ewma import EwmaBandwidthEstimator
import time
from env_hd_erp import Environment
from args import Args, LSTMPredict


class ERP:
    def __init__(self):
        self.estimator = EwmaBandwidthEstimator(8, 3, 10)
        self.default_action = 5

    def action(self, bandwidth, delay, vp_sizes, ad_sizes, out_sizes):
        self.estimator.sample(delay, bandwidth)
        bandwidth_prediction = self.estimator.get_estimate()
        video_sizes = [0, 0, 0, 0, 0, 0]
        quality = 5
        for q in range(6):
            video_sizes[q] = vp_sizes[q] + ad_sizes[q] + out_sizes[q]
            if video_sizes[q] <= bandwidth_prediction:
                quality = q
                break

        # print('bandwidth: ', bandwidth)
        # print('delay: ', delay)
        # print('vp: ', vp_sizes)
        # print('ad: ', ad_sizes)
        # print('out: ', out_sizes)
        return quality


def test(rank, args,
         all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit):

    env = Environment(args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit, random_seed=args.seed + rank)
    model = ERP()
    action = model.default_action
    bandwidth, delay, vp_sizes, ad_sizes, out_sizes, done, (rebuf, cv, blank_ratio, reward, real_vp_bitrate, smooth) = \
        env.step(action)
    state_time = time.time()
    episode_length = 0
    # log = open('new-result-1/test-vp-log20000.txt', 'w')
    # log = open('results-3/log20000.txt', 'w')
    # log = open('train_norway_result-2/test_log3000.txt', 'w')
    log = open('result-1/log-hd-erp.txt', 'w')
    while True:
        episode_length += 1
        action = model.action(bandwidth, delay, vp_sizes, ad_sizes, out_sizes)
        bandwidth, delay, vp_sizes, ad_sizes, out_sizes, done, (rebuf, cv, blank_ratio, reward, real_vp_bitrate, smooth) \
            = env.step(action)
        update = True
        if update:
            print("Time {}, action ({},{},{}), bitrate {:.3f}, rebuf {:.3f}, cv {:.3f}, smooth {:.3f}, reward {:.3f}, episode {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - state_time)),
                action, action, action, real_vp_bitrate, rebuf, cv, smooth,
                reward, episode_length))
            log.write('action: ' + str(1) + ' (' + str(action) + ',' + str(action) + ',' + str(action)
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




