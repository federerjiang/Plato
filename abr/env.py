import numpy as np

import abr.viewport_to_tile as vp_convert
import predictor.lstm as lstm

S_INFO = 6  # bit_rate, buffer_size, next_seg_size, bandwidth_measurement(throughput and time), seg_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 50
VIDEO_SEG_LEN = 1000.0  # milliseconds, every time add this to buffer
BITRATE_LEVELS = 5
TOTAL_VIDEO_SEG = 237  # number of segments
TILE_COLUMN = 12  # number of columns in a segment
TILE_ROW = 6  # number of rows in a segment
BUFFER_THRESH = 10.0 * TILE_COLUMN * TILE_ROW * MILLISECONDS_IN_SECOND  # milliseconds, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # milliseconds, ???
PACKET_PAYLOAD_PORTION = 0.95  #
LINK_RTT = 80  # milliseconds
NOISE_LOW = 0.9  #
NOISE_HIGH = 1.1  #
VIDEO_SIZE_FILE = '../datasets/video/video_size_'
VP_HISTORY_LENGTH = 30  # last 30 frame's viewport points


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_vp_time = all_vp_time
        self.all_vp_unit = all_vp_unit

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_seg_counter = 0
        self.buffer_size = 0
        self.tile_count = 0  # the number of tiles in a downloaded segment

        # pick a random viewport trace file
        self.vp_idx = np.random.randint(len(self.all_vp_time))
        self.vp_time = self.all_cooked_time[self.vp_idx]
        self.vp_unit = self.all_cooked_bw[self.vp_idx]
        # self.vp_sim_ptr = np.random.randint(35, len(self.vp_time))
        self.vp_sim_ptr = 40
        self.last_vp_time = self.vp_time[self.vp_sim_ptr]
        self.vp_history = self.__init_vp_history()

        # pick a random bandwidth trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the bandwidth trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = self.__get_video_size()  # in bytes

    def __init_vp_history(self):
        # vp_history = [x[:] for x in [[0.0] * 4] * VP_HISTORY_LENGTH]
        # pick a random viewport trace file
        self.vp_idx = np.random.randint(len(self.all_vp_time))
        self.vp_time = self.all_cooked_time[self.vp_idx]
        self.vp_unit = self.all_cooked_bw[self.vp_idx]
        # self.vp_sim_ptr = np.random.randint(35, len(self.vp_time))
        self.vp_sim_ptr = 40
        self.last_vp_time = self.vp_time[self.vp_sim_ptr]
        vp_history = self.vp_unit[self.vp_sim_ptr - VP_HISTORY_LENGTH: self.vp_sim_ptr]
        return vp_history

    @staticmethod
    def __get_video_size():
        video_size = {}
        for bitrate in range(BITRATE_LEVELS):
            video_size[bitrate] = {}
            for seg in range(TOTAL_VIDEO_SEG):
                video_size[bitrate][seg] = []
                # each file contains size information of tiles in a segment
                with open(VIDEO_SIZE_FILE + str(bitrate) + '_' + str(seg)) as f:
                    for line in f:
                        video_size[bitrate][seg].append(int(line.split()[0]))

        return video_size

    def __get_download_seg_size(self, vp_quality, ad_quality, out_quality):

        assert vp_quality >= 0
        assert ad_quality >= 0
        assert out_quality >= 0
        assert vp_quality < BITRATE_LEVELS
        assert ad_quality < BITRATE_LEVELS
        assert out_quality < BITRATE_LEVELS

        vp_tiles, ad_tiles, out_tiles = self.get_viewport()
        vp_tiles_size = self.video_size[vp_quality][self.video_seg_counter][vp_tiles]
        ad_tiles_size = self.video_size[ad_quality][self.video_seg_counter][ad_tiles]
        out_tiles_size = self.video_size[out_quality][self.video_seg_counter][out_tiles]
        return vp_tiles_size + ad_tiles_size + out_tiles_size

    def get_viewport(self):
        self.vp_sim_ptr = self.last_vp_time / 0.02  # each frame is about 0.02s
        if self.last_vp_time > self.vp_time[-1] or self.vp_sim_ptr >= len(self.vp_time):
            #  select a new viewport trace file
            self.vp_history = self.__init_vp_history()
        else:
            self.vp_history = self.vp_unit[self.vp_sim_ptr - VP_HISTORY_LENGTH: self.vp_sim_ptr]

        predicted_unit = lstm.predictor(self.vp_history)
        rotation = vp_convert.unit_to_rotation(predicted_unit)
        yaw = rotation[1]
        pitch = rotation[2]
        tile_map = vp_convert.rotation_to_tile(yaw, pitch, 12, 6, 110, 90, 120, 100)
        return tile_map

    def get_video_seg(self, vp_quality, ad_quality, out_quality):
        video_seg_size = self.__get_download_seg_size(vp_quality, ad_quality, out_quality)

        # use the delivery opportunity of mahimahi
        delay = 0.0  # ms
        video_seg_counter_sent = 0.0  # in bytes

        while True:  # download video seg over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE  # in bytes
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_seg_counter_sent + packet_payload > video_seg_size:
                fractional_time = (video_seg_size - video_seg_counter_sent) / throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_seg_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new tiles in a seg
        self.buffer_size += VIDEO_SEG_LEN

        # sleep if the buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the buffer size return to the controller
        return_buffer_size = self.buffer_size

        self.video_seg_counter += 1
        video_seg_remain = TOTAL_VIDEO_SEG - self.video_seg_counter

        end_of_video = False
        if self.video_seg_counter >= TOTAL_VIDEO_SEG:
            end_of_video = True
            self.buffer_size = 0
            self.video_seg_counter = 0

            # pick a random bandwidth trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the bandwidth trace
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.last_vp_time += (delay + sleep_time) / MILLISECONDS_IN_SECOND  # vp time in seconds

        next_video_seg_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_seg_sizes.append(self.video_size[i][self.video_seg_counter])

        # the observed states for the abr agent, and other information to calculate the reward
        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_seg_size, \
            next_video_seg_sizes, \
            end_of_video, \
            video_seg_remain

    # @staticmethod
    # def reset():
    #     return np.zeros((S_INFO, S_LEN))


if __name__ == '__main__':
    test_env = Environment()




