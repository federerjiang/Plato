import torch
import torch.nn as nn
import torch.autograd as autograd
from predictor.data_loader import TEST_LABEL_LENGTH  # in mac
from predictor.data_loader import TestDataLoader
from predictor.data_loader import TrainDataLoader
from predictor.average import average
from predictor.lr import lr
from predictor.lstm import LSTMPredict


LSTM_MODEL_PATH = 'lstm-128-1.model'


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
        # inputs = train_data[i: i+30]
        inputs = torch.FloatTensor(inputs).view(30, 1, -1)
        inputs = autograd.Variable(inputs)
        # label = torch.FloatTensor(train_data[i+30])
        label = autograd.Variable(torch.FloatTensor(label[0]))
        output = model(inputs)
        loss = loss_function(output[-1], label)
        loss_sum += loss
        count += 1
        print(loss)
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
        print(loss)
    return loss_sum / count

if __name__ == '__main__':
    test_data_loader = TrainDataLoader()
    # loss = validate_lstm(torch.load(LSTM_MODEL_PATH), test_data_loader)
    # loss = validate_others(average, test_data_loader)
    loss = validate_others(lr, test_data_loader)
    print(loss)
