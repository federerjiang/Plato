import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loader import TEST_LABEL_LENGTH
from data_loader import TestDataLoader
from average import average
from lr import lr
from cuda_lstm import LSTMPredict


def lstm_predict(model_path, num_layers, hidden_size, inputs, length=TEST_LABEL_LENGTH):

    def init_hidden(num_layers, hidden_size):
        hx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        cx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        hidden = (Variable(hx, volatile=True), Variable(cx, volatile=True))  # convert to Variable as late as possible
        return hidden

    model = torch.load(model_path, map_location='cpu')
    model.hidden = init_hidden(num_layers, hidden_size)

    outputs = []
    inputs = torch.FloatTensor(inputs).view(1, 30, 4)
    inputs = Variable(inputs, volatile=True)
    for _ in range(length):
        output = model(inputs)
        outputs.append(output[-1].data.numpy().tolist())
        t = Variable(torch.randn(1, 30, 4), volatile=True)
        t[:, 0:29, :] = inputs[:, 1:30, :]
        t[:, 29, :] = output.view(1, 30, 4)[:, 29, :]
        inputs = t

    return outputs


def validate_lstm_predict(test_data_loader):
    loss_function = nn.MSELoss()
    loss_sum = 0.0
    count = 0
    for inputs, label in test_data_loader:
        predicts = lstm_predict('adam-lstm-128-2.model', 2, 128, inputs)
        predicts = torch.FloatTensor(predicts)
        # for i in predicts:
        #     print(i)
        predicts = Variable(predicts, volatile=True)
        label = torch.FloatTensor(label)
        label = Variable(label, volatile=True)
        loss = loss_function(predicts, label)
        loss_sum += loss
        count += 1
        print(loss)
        if count == 10000:
            print('validate lstm predict ' + str(TEST_LABEL_LENGTH) + ' frames')
            print(loss_sum / count)
            break


def other_predict(model, inputs, length=TEST_LABEL_LENGTH):
    outputs = []
    for _ in range(length):
        output = model(inputs)
        outputs.append(output)
        inputs.pop(0)
        inputs.append(output)
    print(len(outputs))
    return outputs


def validate_other_predict(model, test_data_loader):
    loss_function = nn.MSELoss()
    loss_sum = 0.0
    count = 0
    for inputs, label in test_data_loader:
        predicts = other_predict(model, inputs)
        predicts = torch.FloatTensor(predicts)
        # for i in predicts:
        #     print(i)
        predicts = Variable(predicts, volatile=True)
        label = torch.FloatTensor(label)
        label = Variable(label, volatile=True)
        loss = loss_function(predicts, label)
        loss_sum += loss
        count += 1
        print(loss)
        if count == 100000:
            print('validate other predict ' + str(TEST_LABEL_LENGTH) + ' frames')
            print(loss_sum / count)
            break


if __name__ == "__main__":
    test_data_loader = TestDataLoader()
    # validate_lstm_predict(test_data_loader)

    # print('average results ')
    # validate_other_predict(average, test_data_loader)
    print('lr results')
    validate_other_predict(lr, test_data_loader)




