class Args:
    def __init__(self):
        self.s_info = 11  # bit_rate, buffer_size, next_seg_size, bandwidth_measurement(throughput and time), seg_til_video_en
        self.s_len = 8  # take how many frames in the past
        self.a_dim = 125  # or 25
        self.vp_window_len = 30
        self.milliseconds_in_second = 1000.0
        self.b_in_mb = 1000000.0
        self.bits_in_byte = 8.0
        self.random_seed = 50
        self.video_seg_len = 1000.0  # milliseconds, every time add this to buffer
        self.qp_levels = 5
        self.total_video_seg = 237
        self.tile_column = 12
        self.tile_row = 6
        self.buffer_thresh = 3000.0  # max video length in buffer (3 segments)
        self.buffer_norm_factor = 3.0  # used for reward function
        self.quality_penalty = 1
        self.rebuf_penalty = 6
        self.smooth_penalty = 1.0
        self.cv_penalty = 1.0
        self.blank_penalty = 1.0
        self.drain_buffer_sleep_time = 500.0  # milliseconds
        self.packet_payload_portion = 0.95
        self.link_rtt = 80  # milliseconds
        self.noise_low = 0.9
        self.noise_high = 1.1
        self.video_size_file = '../../datasets/video/video_size_'
        self.vp_history_len = 30  # use last 30 frames' viewport to predict
        self.predictor_path = 'adam-lstm-128-1.model'
        self.tile_column = 12
        self.tile_row = 6
        self.vp_length = 110
        self.vp_height = 90
        self.ad_length = 170
        self.ad_height = 110
        self.video_qp = [10, 15, 20, 25, 30]  # need to calculate
        self.video_bitrate = [180, 150, 120, 90, 60]  # for each tile, in Kbp/s

        # model training parameters
        self.batch_size = 1280
        self.num_mini_batch = 20
        self.lr = 5e-5
        self.a_lr = 3e-4
        self.c_lr = 1e-4
        self.gamma = 0.99
        self.gae = 0.95
        self.clip = 0.2
        self.ent_coef = 0.
        self.lr = 1e-4
        self.gamma = 0.99
        self.tau = 1.00
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 50
        self.seed = 1
        self.num_processes = 1
        self.num_steps = 20000
        self.max_episode_length = 1000000
        self.seed = 30


BATCH_SIZE = 32
SEQ_LEN = 30
TAG_SIZE = 3
CUDA = True

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F


class LSTMPredict(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, tag_size=TAG_SIZE, use_cuda=CUDA):
        super(LSTMPredict, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tag_size = tag_size
        self.use_cuda = use_cuda

        # self.in2lstm = nn.Linear(tag_size, input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.init_lstm()

        self.lstm2tag = nn.Linear(self.hidden_size, self.tag_size)
        # nn.init.normal(self.lstm2tag.weight)

        self.hidden = self.init_hidden()  # initial hidden state for LSTM network

    def init_lstm(self):
        for name, weights in self.lstm.named_parameters():
            if len(weights.data.shape) == 2:
                nn.init.kaiming_normal(weights)
            if len(weights.data.shape) == 1:
                nn.init.normal(weights)

    def init_hidden(self):
        hx = torch.nn.init.xavier_normal(torch.randn(self.num_layers, BATCH_SIZE, self.hidden_size))
        cx = torch.nn.init.xavier_normal(torch.randn(self.num_layers, BATCH_SIZE, self.hidden_size))
        if self.use_cuda:
            hx, cx = hx.cuda(), cx.cuda()
        hidden = (autograd.Variable(hx), autograd.Variable(cx))  # convert to Variable as late as possible
        return hidden

    def forward(self, orientations):
        # orientation_seq is a 3 dimensional tensor with shape [batch_size, seq_len, tag_size]
        # lstm_in is a 2 dimensional tensor with shape [seq_len, input_size]
        # inputs is a 3 dimensional tensor with shape [batch_size, seq_len, tag_size]
        lstm_out, self.hidden = self.lstm(orientations, self.hidden)
        # print(lstm_out.size())
        tag_scores = F.tanh(self.lstm2tag(lstm_out.contiguous().view(-1, self.hidden_size)))
        # print(tag_scores.size())
        return tag_scores.view(-1, self.tag_size)

