import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import TrainDataLoader  # in ubuntu
from data_loader import CudaTrainLoader  # in ubuntu
import math

BATCH_SIZE = 32
SEQ_LEN = 30
TAG_SIZE = 3
CUDA = True


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


def train_model(model, learning_rate, data_loader, epoch=10, count_max=10000):
    use_cuda = torch.cuda.is_available()
    print('cuda: ' + str(use_cuda))

    model.train()  # for cuda speed up
    if use_cuda:
        model = model.cuda()

    loss_function = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for poch in range(epoch):
        count = 0
        loss_avg = 0.0
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        learning_rate *= 0.2
        for inputs, label in data_loader:
            # inputs = train_data[i: i+30]
            if count == count_max:
                break
            inputs = torch.FloatTensor(inputs)
            label = torch.FloatTensor(label)
            if use_cuda:
                inputs, label = inputs.cuda(), label.cuda()
                # print('change inputs, label to cuda type')

            inputs = autograd.Variable(inputs)
            label = autograd.Variable(label).view(-1, TAG_SIZE)

            model.zero_grad()
            model.hidden = model.init_hidden()

            output = model(inputs)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            loss_avg += loss
            count += 1
            if count % 1000 == 0:
                print(poch, count, loss_avg / 1000)
                loss_avg = 0.0
        # print(poch, loss_avg / count_max)


def try_hyper_para(hidden_size_list, num_layer_list, data_loader, epoc, count_max):
    for hidden_size in hidden_size_list:
        for num_layers in num_layer_list:
            model = LSTMPredict(input_size=TAG_SIZE, hidden_size=hidden_size, num_layers=num_layers, tag_size=TAG_SIZE)
            train_model(model, learning_rate=0.001, data_loader=data_loader, epoch=epoc, count_max=count_max)
            print("finished training")
            model_name = 'adam-lstm-' + str(hidden_size) + '-' + str(num_layers) + '.model'
            # loss_name = 'loss-' + str(hidden_size) + '-' + str(num_layers) + '.dat'
            torch.save(model, model_name)
            print('saved model: ' + model_name)
            # save_loss(losses, loss_name)
            # print('saved loss data: ' + loss_name)

if __name__ == "__main__":
    data_loader = TrainDataLoader()
    cuda_data_loader = CudaTrainLoader(data_loader)
    # hidden_size_list = [128, 256, 512]
    # num_layer_list = [1, 2]
    hidden_size_list = [128, 256]
    num_layer_list = [1, 2]
    try_hyper_para(hidden_size_list, num_layer_list, cuda_data_loader, epoc=10, count_max=30000)
