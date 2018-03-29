import numpy as np
import torch
from torch.autograd import Variable
import math

# from args import Args
from abr.ppo.args import Args, LSTMPredict


class Environment:
    def __init__(self, args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit):
        np.random.seed(args.random_seed)
        self.args = args

        self.all_vp_time = all_vp_time
        self.all_vp_unit = all_vp_unit
        self.vp_window_len = args.vp_window_len

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_seg_counter = 0
        self.buffer_size = 0

        # pick a random viewport trace file, and a random start point in a file
        self.vp_idx = np.random.randint(len(self.all_vp_time))
        self.vp_time = self.all_vp_time[self.vp_idx]
        self.vp_unit = self.all_vp_unit[self.vp_idx]
        self.vp_sim_ptr = np.random.randint(40, len(self.vp_time)-100)
        self.vp_sim_ptr_max = len(self.vp_time) - 100
        # self.last_vp_time = self.vp_time[self.vp_sim_ptr]
        self.vp_history = self._init_vp_history()
        self.vp_real_future = self._real_vp_future()
        self.vp_playback_time = 0
        self.vp_predictor = self._load_predictor(args)
        self.vp_preq_future = self._pred_vp_future()

        # pick a random bandwidth trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        # randomize the start point of the bandwidth trace
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = self._get_video_size(args)  # in bytes, each tile size in a segment
        # self.vp_sizes = np.zeros(5)
        # self.ad_sizes = np.zeros(5)
        # self.out_sizes = np.zeros(5)
        # self.pred_tile_map = [x[:] for x in [[0] * args.tile_column] * args.tile_row]
        # self.real_tile_map = [x[:] for x in [[0] * args.tile_column] * args.tile_row]
        self.pred_tile_map = self._update_tile_map([[0.0, 0.0, 0.0]])
        self.real_tile_map = self._update_tile_map([[0.0, 0.0, 0.0]])
        self.action_map = self._set_action_map()
        self.vp_sizes, self.ad_sizes, self.out_sizes = self._get_tile_area_sizes()
        self.state = np.zeros((args.s_info, args.s_len))

        self.last_real_vp_bitrate = 1  # should be the default quality for viewport

    def _init_vp_history(self):
        self.vp_idx = np.random.randint(len(self.all_vp_time))
        self.vp_time = self.all_vp_time[self.vp_idx]
        self.vp_unit = self.all_vp_unit[self.vp_idx]
        self.vp_sim_ptr = np.random.randint(40, len(self.vp_time)-100)
        self.vp_sim_ptr_max = len(self.vp_time) - 100
        # self.last_vp_time = self.vp_time[self.vp_sim_ptr]
        self.vp_playback_time = 0
        vp_history = self.vp_unit[self.vp_sim_ptr - self.vp_window_len: self.vp_sim_ptr]
        return vp_history

    def _update_vp_history(self, delay, rebuffer, sleep):
        self.vp_playback_time = delay - rebuffer + sleep
        self.vp_sim_ptr += math.floor(self.vp_playback_time / 33.3)
        if self.vp_sim_ptr <= self.vp_sim_ptr_max:
            self.vp_history = self.vp_unit[self.vp_sim_ptr - self.vp_window_len: self.vp_sim_ptr]
        else:
            self.vp_history = self._init_vp_history()

    def _real_vp_future(self):
        buffer_frame_len = math.floor(self.buffer_size / 33.3)
        start = self.vp_sim_ptr + buffer_frame_len
        end = start + 30
        # print('real vp future', buffer_frame_len, start, end)
        return self.vp_unit[start: end]

    @staticmethod
    def _load_predictor(args):
        def init_hidden(num_layers, hidden_size):
            hx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
            cx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
            hidden = (Variable(hx, volatile=True), Variable(cx, volatile=True))  # convert to Variable as late as possible
            return hidden
        model = torch.load(args.predictor_path, map_location='cpu')
        model.hidden = init_hidden(1, 128)
        return model

    def _pred_vp_future(self):
        # predict the next segment needed to be downloaded
        def lstm_predict(model, inputs, length):
            inputs = torch.FloatTensor(inputs).view(1, 30, 3)
            inputs = Variable(inputs, volatile=True)
            outputs = []
            for _ in range(length):
                output = model(inputs)
                outputs.append(output[-1].data.numpy().tolist())
                t = Variable(torch.randn(1, 30, 3), volatile=True)
                t[:, 0:29, :] = inputs[:, 1:30, :]
                t[:, 29, :] = output.view(1, 30, 3)[:, 29, :]
                inputs = t
            return outputs
        model = self.vp_predictor
        buffer_frame_len = math.floor(self.buffer_size / 33.3)
        outputs = lstm_predict(model, self.vp_history, buffer_frame_len+30)
        return outputs[buffer_frame_len: buffer_frame_len+30]

    def _update_tile_map(self, vp_future):
        def rotation_to_vp_tile(yaw, pitch, tile_column, tile_row, vp_length, vp_height, tile_map, tag):
            tile_length = 360 / tile_column
            tile_height = 180 / tile_row

            vp_pitch = pitch + 90
            vp_up = vp_pitch + vp_height / 2
            if vp_up > 180:
                vp_up = 179
            vp_down = vp_pitch - vp_height / 2
            if vp_down < 0:
                vp_down = 0

            vp_yaw = yaw + 180
            vp_part = 1
            vp_left = vp_yaw - vp_length / 2
            vp_right = vp_yaw + vp_length / 2
            if vp_left < 0:
                vp_part = 2
                vp_left_1 = 0
                vp_left_2 = vp_left + 360
                vp_right_1 = vp_right
                vp_right_2 = 359
            if vp_right > 360:
                vp_part = 2
                vp_right_1 = vp_right - 360
                vp_right_2 = 359
                vp_left_1 = 0
                vp_left_2 = vp_left

            def get_tiles(left, right, up, down, tag):
                col_start = math.floor(left / tile_length)
                col_end = math.floor(right / tile_length)
                row_start = math.floor(down / tile_height)
                row_end = math.floor(up / tile_height)
                count = 0
                for row in range(row_start, row_end + 1):
                    for col in range(col_start, col_end + 1):
                        count += 1
                        if tile_map[row][col] != 1:  # if tile is not vp, then is set tag
                            tile_map[row][col] = tag
                return count

            tile_count = 0
            if vp_part == 1:
                tile_count = get_tiles(vp_left, vp_right, vp_up, vp_down, tag)
            if vp_part == 2:
                tile_count = get_tiles(vp_left_1, vp_right_1, vp_up, vp_down, tag)
                tile_count += get_tiles(vp_left_2, vp_right_2, vp_up, vp_down, tag)

            # print(tile_count)
            # return tile_map
        args = self.args
        tile_map = [x[:] for x in [[0] * args.tile_column] * args.tile_row]
        for rotation in vp_future:  # set vp tile tag
            pitch = rotation[1] * 180 / math.pi
            yaw = rotation[2] * 180 / math.pi
            rotation_to_vp_tile(yaw, pitch, args.tile_column, args.tile_row, args.vp_length, args.vp_height,
                                tile_map, 1)
        for rotation in vp_future:  # set ad tile tag
            pitch = rotation[1]
            yaw = rotation[2]
            rotation_to_vp_tile(yaw, pitch, args.tile_column, args.tile_row, args.ad_length, args.ad_height,
                                tile_map, 2)
        return tile_map

    def _get_tile_area_sizes(self):
        vp_sizes = []
        ad_sizes = []
        out_sizes = []
        seg = self.video_seg_counter
        args = self.args
        for qp in range(args.qp_levels):
            vp_sum = 0
            ad_sum = 0
            out_sum = 0
            for row in range(args.tile_row):
                for column in range(args.tile_column):
                    if self.pred_tile_map[row][column] == 1:
                        vp_sum += self.video_size[qp][seg][row * args.tile_column + column]
                    elif self.pred_tile_map[row][column] == 2:
                        ad_sum += self.video_size[qp][seg][row * args.tile_column + column]
                    else:
                        out_sum += self.video_size[qp][seg][row * args.tile_column + column]
            vp_sizes.append(vp_sum)
            ad_sizes.append(ad_sum)
            out_sizes.append(out_sum)
        return vp_sizes, ad_sizes, out_sizes

    @staticmethod
    def _get_video_size(args):
        video_size = {}
        for qp in range(args.qp_levels):
            video_size[qp] = {}
            for seg in range(args.total_video_seg):
                video_size[qp][seg] = []
                with open(args.video_size_file + str(qp) + '_' + str(seg)) as f:
                    for line in f:
                        video_size[qp][seg].append(int(line.split()[0]))
        return video_size

    @staticmethod
    def _set_action_map():
        vp_levels = [0, 1, 2, 3, 4]
        ad_levels = out_levels = [-1, 0, 1, 2, 3, 4]
        action_map = []
        for vp in range(len(vp_levels)):
            for ad in range(len(ad_levels)):
                for out in range(len(out_levels)):
                    action_map.append((vp_levels[vp], ad_levels[ad], out_levels[out]))
        return action_map

    def _get_states_rewards(self, vp, ad, out):
        # get the average real-viewport quality according to the downloaded three area qualities.
        args = self.args
        count = [0, 0, 0]
        for row in range(args.tile_row):
            for column in range(args.tile_column):
                if self.real_tile_map[row][column] == 1:
                    count[self.pred_tile_map[row][column]] += 1
        out_count, vp_count, ad_count = count[0], count[1], count[2]
        vp_bitrate = self.args.video_bitrate[vp]
        ad_bitrate = self.args.video_bitrate[ad] if ad >= 0 else 0
        out_bitrate = self.args.video_bitrate[out] if out >= 0 else 0
        real_vp_bitrate = vp_count * vp_bitrate + ad_count * ad_bitrate + out_count * out_bitrate
        print(vp_count, ad_count, out_count)

        # get accuracy
        total_count = vp_count + ad_count + out_count
        vp_acc = vp_count / total_count
        ad_acc = ad_count / total_count
        out_acc = out_count / total_count

        # get spatial quality variance
        mean = real_vp_bitrate / total_count
        sum_pow = vp_count * ((vp_bitrate - mean)**2) + ad_count * ((ad_bitrate - mean)**2) + out_count * ((out_bitrate - mean)**2)
        std = math.sqrt(sum_pow / (total_count - 1))
        cv = std / mean

        # get blank area percentage
        blank_count = 0
        if ad_bitrate == 0:
            blank_count += ad_count
        if out_bitrate == 0:
            blank_count += out_count
        blank_ratio = blank_count / total_count

        return real_vp_bitrate / 1000, vp_acc, ad_acc, out_acc, cv, blank_ratio

    def step(self, action):
        vp_quality, ad_quality, out_quality = self.action_map[action]
        vp_size = self.vp_sizes[vp_quality]
        ad_size = self.ad_sizes[ad_quality] if ad_quality >= 0 else 0
        out_size = self.out_sizes[out_quality] if out_quality >= 0 else 0
        video_seg_size = vp_size + ad_size + out_size
        # print(vp_quality, ad_quality, out_quality)
        # print(vp_size/1000000, ad_size/1000000, out_size/1000000)
        # use the delivery opportunity of mahimahi
        delay = 0.0  # seconds
        video_seg_counter_sent = 0.0  # bytes

        while True:  # download video seg over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] * self.args.b_in_mb / self.args.bits_in_byte  # bytes / s
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time  # seconds
            # print('throughput', throughput/1000000)
            packet_load = throughput * duration * self.args.packet_payload_portion

            if video_seg_counter_sent + packet_load > video_seg_size:
                fractional_time = (video_seg_size - video_seg_counter_sent) / throughput / self.args.packet_payload_portion
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_seg_counter_sent += packet_load
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= self.args.milliseconds_in_second  # ms
        delay += self.args.link_rtt
        # add a multiplicative noise to the delay
        delay *= np.random.uniform(self.args.noise_low, self.args.noise_high)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new seg in the buffer
        self.buffer_size += self.args.video_seg_len

        # sleep if the buffer gets too large
        sleep_time = 0
        if self.buffer_size > self.args.buffer_thresh:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - self.args.buffer_thresh
            sleep_time = np.ceil(drain_buffer_time / self.args.drain_buffer_sleep_time) * self.args.drain_buffer_sleep_time
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
                if duration > sleep_time / self.args.milliseconds_in_second:
                    self.last_mahimahi_time += sleep_time / self.args.milliseconds_in_second
                    break
                sleep_time -= duration * self.args.milliseconds_in_second
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the buffer size return to the controller
        # return_buffer_size = self.buffer_size

        self.video_seg_counter += 1
        video_seg_remain = self.args.total_video_seg - self.video_seg_counter
        # print('video seg remain:', video_seg_remain)
        done = False
        if self.video_seg_counter >= self.args.total_video_seg:
            done = True
            self.reset()

        self._update_vp_history(delay=delay, rebuffer=rebuf, sleep=sleep_time)
        pred_vp_future = self._pred_vp_future()
        # print('pred vp future:', pred_vp_future)
        self.pred_tile_map = self._update_tile_map(pred_vp_future)
        # print('pred tile map', self.pred_tile_map)
        self.vp_sizes, self.ad_sizes, self.out_sizes = self._get_tile_area_sizes()  # next seg sizes of all qualities

        real_vp_future = self._real_vp_future()
        # print(real_vp_future)
        self.real_tile_map = self._update_tile_map(real_vp_future)
        # print('real time map:', self.real_tile_map)
        real_vp_bitrate, vp_acc, ad_acc, out_acc, cv, blank_ratio = self._get_states_rewards(vp_quality, ad_quality,
                                                                                             out_quality)

        # dequeue history record
        self.state = np.roll(self.state, -1, axis=1)
        # this should be S_INFO number of terms
        self.state[0, -1] = real_vp_bitrate  # last quality (not normalized !!)
        self.state[1, -1] = self.buffer_size / 1000 / self.args.buffer_norm_factor  # (buffer size)
        self.state[2, -1] = float(video_seg_size) / float(delay) / 1000  # kilo byte / ms (throughput)
        self.state[3, -1] = float(delay) / 1000 / self.args.buffer_norm_factor  # (delay time s)
        self.state[4, :self.args.qp_levels] = np.array(self.vp_sizes) / 1000 / 1000  # mega byte (vp)
        self.state[5, :self.args.qp_levels] = np.array(self.ad_sizes) / 1000 / 1000  # mega byte (ad)
        self.state[6, :self.args.qp_levels] = np.array(self.out_sizes) / 1000 / 1000  # mega byte (out)
        self.state[7, -1] = video_seg_remain / self.args.total_video_seg
        self.state[8, -1] = vp_acc  # pred_vp accuracy
        self.state[9, -1] = ad_acc  # pred_ad accuracy
        self.state[10, -1] = out_acc  # pred_out accuracy

        # reward is video quality (Mbps) - rebuffer penalty - smooth penalty - spatial variance - blank tiles percentage
        # the reward function is not complete now, needed to be modified later
        reward = real_vp_bitrate \
                 - self.args.rebuf_penalty * rebuf / 1000 \
                 - self.args.smooth_penalty * np.abs(real_vp_bitrate - self.last_real_vp_bitrate) \
                 - self.args.cv_penalty * cv \
                 - self.args.blank_penalty * blank_ratio

        self.last_real_vp_bitrate = real_vp_bitrate

        # print('buffer size:', self.buffer_size)
        print('rebuffer time:', rebuf)
        print('cv', cv)
        print('blank', blank_ratio)
        # print('delay', delay)
        print('real vp bitrate:', real_vp_bitrate)
        print('reward:', reward)

        # print('state:', self.state)
        print('')
        return self.state, reward, done

    def reset(self):
        self.buffer_size = 0
        self.video_seg_counter = 0
        self.last_real_vp_bitrate = 1
        self.vp_playback_time = 0

        # pick a random bandwidth trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the bandwidth trace
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.state = np.zeros((self.args.s_info, self.args.s_len))
        self.vp_history = self._init_vp_history()
        self.pred_tile_map = self._update_tile_map([[0.0, 0.0, 0.0]])
        self.real_tile_map = self._update_tile_map([[0.0, 0.0, 0.0]])

    @staticmethod
    def sample_action():
        return np.random.randint(0, 150)


# test env adn ppo model
if __name__ == "__main__":
    from abr.load_bw_traces import load_trace
    from abr.load_viewport_trace import load_viewport_unit
    bw_trace_folder = '../../datasets/bw_trace/sim_belgium/'
    vp_trace_folder = '../../datasets/viewport_trace/new_cooked_train_dataset/'
    args = Args()

    all_cooked_time, all_cooked_bw, _ = load_trace(bw_trace_folder)
    all_vp_time, all_vp_unit = load_viewport_unit(vp_trace_folder)
    # print(len(all_cooked_bw), len(all_cooked_time))
    # print(len(all_vp_time), len(all_vp_unit))
    env = Environment(args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit)
    # env.step(action=144)
    # env.step(144)
    from torch.autograd import Variable
    # fc = torch.nn.Linear(1, 128)
    # conv = torch.nn.Conv1d(1, 128, 3)
    # actor_linear = torch.nn.Linear(512, 180)
    from abr.ppo.model import ActorModel, CriticModel
    actor = ActorModel()
    critic = CriticModel()
    print(actor)
    tag = 1
    while tag > 0:
        action = env.sample_action()
        print('action', action)
        state, reward, done = env.step(action)
        print(state.shape)
        inputs = Variable(torch.FloatTensor([state, state]).view(-1, 11, 8))
        # logit = actor(inputs)
        # print(logit.shape)
        v = critic(inputs)
        print(v)
        # print(inputs.shape)
        # out = fc(inputs[:, 0:1, -1])
        # print('linear:', out.shape)
        # out_conv = conv(inputs[:, 2:3, :5])
        # out = out.view(-1, 1, 128)
        # print(out.shape)
        # # print(out_conv.shape)
        # out_conv = out_conv.view(1, -1, 128)
        # print(out_conv.shape)
        # merge = torch.cat((out, out_conv), dim=1)
        # merge = merge.view(1, -1)
        # print(merge.shape)
        # logit = actor_linear(merge)
        # print(logit)

        tag -= 1












