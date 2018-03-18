import torch
import torch.nn as nn
import torch.autograd as autograd
from data_loader import TEST_LABEL_LENGTH  # in mac
from data_loader import TestDataLoader
from data_loader import TrainDataLoader
from average import average
from lr import lr
from mp_lstm import LSTMPredict


LSTM_MODEL_PATH = 'lstm-256-1.model'


def lstm_predict(test_sample):
    model = torch.load(LSTM_MODEL_PATH)
    outputs = []
    for _ in range(TEST_LABEL_LENGTH):
        output = model(test_sample)
        outputs.append(output)
        test_sample


def validate_lstm(model, test_data_loader):
    loss_function = nn.MSELoss()
    loss_sum = 0.0
    count = 0
    for inputs, label in test_data_loader:
        # for i in range(len(inputs)):
        #     print(inputs[i], label[i])
        # inputs = train_data[i: i+30]
        inputs = torch.FloatTensor(inputs).view(30, 1, -1)
        inputs = autograd.Variable(inputs, volatile=True)
        # label = torch.FloatTensor(train_data[i+30])
        # label = autograd.Variable(torch.FloatTensor(label[0]))
        output = model(inputs)
        # output = output.view(-1, 4)
        # loss = loss_function(output[-1], autograd.Variable(torch.FloatTensor(label[-1])))
        loss = loss_function(output[-1], autograd.Variable(torch.FloatTensor(label[0]), volatile=True))
        # print(loss.data)
        # loss_sum += loss
        if count % 1000 == 0:
            print(loss_sum / 1000)
            loss_sum = 0.0
            # break
        else:
            loss_sum += loss
        if count == 100000:
            break
        count += 1

    return loss_sum / count


def validate_others(model, test_data_loader):
    loss_sum = 0.0
    count = 0
    for inputs, label in test_data_loader:
        output = model(inputs)
        label = label[0]
        loss = 0.0
        for k in range(4):
            loss += (output[k] - label[k]) * (output[k] - label[k])
        loss_sum += loss / 4
        count += 1
        if count == 100000:
            # print(loss_sum / count)
            break
        # print(loss)

    return loss_sum / count


def init_hidden(num_layers, hidden_size):
        hx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size)).cuda()
        cx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size)).cuda()
        hidden = (autograd.Variable(hx), autograd.Variable(cx))  # convert to Variable as late as possible
        return hidden


def get_lstm_loss_gpu(model_path, num_layers, hidden_size, test_data_loader):
    batch_model = torch.load(model_path)
    # model.load_state_dict(own_state)
    # batch_model = torch.load(model_path, map_location='cpu')
    batch_model.hidden = init_hidden(num_layers, hidden_size)
    batch_model.lstm.flatten_parameters()
    model = LSTMPredict(input_size=4, hidden_size=hidden_size, num_layers=num_layers, tag_size=4)
    model.load_state_dict(batch_model.state_dict())
    # batch_model.lstm2tag.flatten_parameters()
    loss = validate_lstm(model, test_data_loader)
    print(model_path)
    print(loss)


def get_lstm_loss_cpu(model_path, num_layers, hidden_size, test_data_loader):
    pass


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    test_data_loader = TestDataLoader()
    # batch_model = torch.load('lstm-128-1.model', map_location='cpu')
    # batch_model.hidden = init_hidden(1, 128)
    # batch_model.lstm.flatten_parameters()
    # loss = validate_lstm(batch_model, test_data_loader)
    # print(batch_model.state_dict())
    # own_state = batch_model.state_dict()
    # print(len(own_state))
    # for i in own_state:
    #     print(i)
    # model = LSTMPredict(input_size=4, hidden_size=128, num_layers=1, tag_size=4)
    # model.load_state_dict(own_state)
    # loss = validate_lstm(batch_model, test_data_loader)

    model = torch.load('mp-sgd-lstm-128-1.model')
    loss = validate_lstm(model, test_data_loader)
    print(loss)



    # loss = validate_others(average, test_data_loader)
    # print('average: ')
    # print(loss)
    # loss = validate_others(lr, test_data_loader)
    # print('linear regression ')
    # print(loss)

    # get_lstm_loss('lstm-128-1.model', 1, 128, test_data_loader)
    # get_lstm_loss('adam-lstm-128-1.model', 1, 128)
    # get_lstm_loss('lstm-128-2.model', 2, 128)
    # get_lstm_loss('adam-lstm-128-2.model', 2, 128)
    # get_lstm_loss('lstm-256-1.model', 1, 256)
    # get_lstm_loss('adam-lstm-256-1.model', 1, 256)
    # get_lstm_loss('lstm-512-1.model', 1, 512)

