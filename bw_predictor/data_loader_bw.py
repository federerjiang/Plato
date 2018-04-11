import os
import numpy as np
import torch
from args import Args


BW_TRACE = '../datasets/bw_trace/'
TRAIN_DATASET = BW_TRACE + 'train_sim_traces/'
TRAIN_DATASET_BK = BW_TRACE + 'sim_belgium/'
TEST_DATASET = BW_TRACE + 'test_sim_traces/'


class TrainDataLoader:
    def __init__(self, args, trace_folder=TRAIN_DATASET_BK, random_seed=14):
        np.random.seed(random_seed)

        self.label_sample_length = args.label_sample_length
        self.train_sample_length = args.train_sample_length

        self.trace_folder = trace_folder
        self.all_bw = self._load_bw()
        self.bw_idx = np.random.randint(low=0, high=len(self.all_bw))
        self.bw = self.all_bw[self.bw_idx]
        self.bw_start_max = len(self.bw) - self.label_sample_length - 2
        self.idx = 0

    def _load_bw(self):
        all_bw = []
        for root, dirnames, filenames in os.walk(self.trace_folder):
            for filename in filenames:
                trace_file = os.path.join(root, filename)
                bw = []
                with open(trace_file, 'r') as f:
                    for line in f:
                        parse = line.split()
                        bw.append(float(parse[1]))
                if len(bw) > 50:
                    all_bw.append(bw)
        return all_bw

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= self.bw_start_max:
            self.bw_idx = np.random.randint(low=0, high=len(self.all_bw))
            self.bw = self.all_bw[self.bw_idx]
            self.bw_start_max = len(self.bw) - self.train_sample_length - 2
            self.idx = 0
        train_sample = self.bw[self.idx: self.idx + self.train_sample_length]
        label_sample = self.bw[self.idx + 1: self.idx + 1 + self.train_sample_length]
        return train_sample, label_sample


class CudaTrainLoader:
    def __init__(self, args, train_data_loader):
        self.batch_size = args.batch_size
        self.data_loader = train_data_loader

    def __iter__(self):
        return self

    def __next__(self):
        batch_train_sample = []
        batch_label_sample = []
        count = 0
        for train_sample, label_sample in self.data_loader:
            count += 1
            batch_train_sample.append(train_sample)
            batch_label_sample.append(label_sample)
            if count == self.batch_size:
                break
        return batch_train_sample, batch_label_sample


def transform():
    # import math.exp as exp
    import math
    exp = math.exp
    tanh = lambda x: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    print(tanh(20))
    import torch
    x = torch.FloatTensor([5, 20, 30, 40])
    y = torch.tanh(x)
    # y = torch.sigmoid(x)
    print(y)


if __name__ == '__main__':
    args = Args()
    data_loader = TrainDataLoader(args)
    # count = 0
    # for train, label in data_loader:
        # train = torch.FloatTensor(train)
        # train = torch.tanh(train)
        # print('train', train)
    #     print('label', label)
    #     count += 1
    #     if count % 32 == 0:
    #         print('count', count/32)
    # transform()

    cuda_data_loader = CudaTrainLoader(args, data_loader)
    count = 0
    for train, label in cuda_data_loader:
        print(train.size())
        print(label.size())
        count += 1
        print('count', count)
        # if count == 1:
        #     break
