import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
# from predictor.data_loader import TrainDataLoader  # in mac
from data_loader import TrainDataLoader  # in ubuntu


class LSTMPredict(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, tag_size):
        super(LSTMPredict, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.in2lstm = nn.Linear(tag_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.init_lstm()

        self.lstm2tag = nn.Linear(hidden_size, tag_size)
        # nn.init.normal(self.lstm2tag.weight)

        self.hidden = self.init_hidden()  # initial hidden state for LSTM network

    def init_lstm(self):
        for name, weights in self.lstm.named_parameters():
            if len(weights.data.shape) == 2:
                nn.init.kaiming_normal(weights)
            if len(weights.data.shape) == 1:
                nn.init.normal(weights)

    def init_hidden(self):
        hx = torch.nn.init.xavier_normal(autograd.Variable(torch.randn(self.num_layers, 1, self.hidden_size)))
        cx = torch.nn.init.xavier_normal(autograd.Variable(torch.randn(self.num_layers, 1, self.hidden_size)))
        hidden = (hx, cx)
        return hidden

    def forward(self, orientations):
        # orientation_seq is a 2 dimensional tensor with shape [seq_len, tag_size]
        # lstm_in is a 2 dimensional tensor with shape [seq_len, input_size]
        # inputs is a 3 dimensional tensor with shape [seq_len, 1, -1]
        lstm_out, self.hidden = self.lstm(orientations, self.hidden)
        # print(lstm_out)
        tag_scores = F.tanh(self.lstm2tag(lstm_out.view(-1, self.hidden_size)))
        # print(tag_scores)
        return tag_scores


def train_model(rank, lock, counter, model, data_loader, epoch=10, count_max=10000):
    loss_function = nn.MSELoss()
    batch_loss = 0.0
    for poch in range(epoch):
        count = 0
        if poch == 0:
            optimizer = optim.SGD(model.parameters(), lr=0.001)
        elif poch == 1:
            optimizer = optim.SGD(model.parameters(), lr=0.0003)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.00001)
        for inputs, label in data_loader:
            # inputs = train_data[i: i+30]
            if count == count_max:
                break
            inputs = torch.FloatTensor(inputs).view(30, 1, -1)
            inputs = autograd.Variable(inputs)
            # print(inputs)
            # label = torch.FloatTensor(train_data[i+1: i+31])
            label = torch.FloatTensor(label)
            label = autograd.Variable(label)
            # print(inputs.size(), label.size())
            model.zero_grad()
            model.hidden = model.init_hidden()
            output = model(inputs)
            sum_loss = 0.0
            for index in range(len(output)):
                sum_loss += loss_function(output[index], label[index])
            loss = sum_loss / len(output)

            with lock:
                counter.value += 1

            loss.backward()
            optimizer.step()
            # print(inputs)
            if count % 1000 == 0:
                batch_loss = batch_loss / 1000
                print(rank, poch, count, batch_loss)
                batch_loss = 0.0
            else:
                batch_loss += loss

            count += 1


def save_loss(losses, path):
    with open(path, "w+") as f:
        for loss in losses:
            f.write(str(loss) + "\n")


def main_train(data_loader, hidden_size, num_layers, num_processes, epoch=1, count_max=100000):
    model = LSTMPredict(input_size=4, hidden_size=hidden_size, num_layers=num_layers, tag_size=4)
    model.share_memory()

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train_model, args=(rank, lock, counter, model, data_loader, epoch, count_max))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def try_hyper_para(hidden_size_list, num_layer_list, data_loader, epoch, count_max):
    for hidden_size in hidden_size_list:
        for num_layers in num_layer_list:
            model = LSTMPredict(input_size=4, hidden_size=hidden_size, num_layers=num_layers, tag_size=4)
            main_train(data_loader, hidden_size=128, num_layers=1, num_processes=15, epoch=epoch, count_max=count_max)
            print("finished training")
            model_name = 'lstm-' + str(hidden_size) + '-' + str(num_layers) + '.model'
            loss_name = 'loss-' + str(hidden_size) + '-' + str(num_layers) + '.dat'
            torch.save(model, model_name)
            print('saved model: ' + model_name)
            save_loss(losses, loss_name)
            print('saved loss data: ' + loss_name)

if __name__ == "__main__":
    data_loader = TrainDataLoader()
    hidden_size_list = [128, 256]
    num_layer_list = [1, 2]
    try_hyper_para(hidden_size_list, num_layer_list, data_loader, epoch=4, count_max=300000)
    # main_train(data_loader, hidden_size=128, num_layers=1, num_processes=15, epoch=4, count_max=300000)

    # model = torch.load("lstm-512-1.model")
    # print(validate(model, train_data))
    # model = torch.load("lstm-256-1.model")
    # print(validate(model, train_data))
    # model = torch.load("lstm-128-2.model")
    # print(validate(model, train_data))
    # print(avg_prediction(train_data))





