import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from args import Args
from data_loader import TrainDataLoader
from data_loader import TestDataLoader


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear(x)
        return out


def convert(inputs):
    '''
    convert inputs arrary from 30x3 to 3x30
    :param inputs:
    :return: a new list
    '''
    outputs = []
    for dim1 in range(3):
        output = []
        for dim2 in range(30):
            output.append(inputs[dim2][dim1])
        outputs.append(output)
    return outputs


def train_model(model, learning_rate, data_loader, epoch=10, count_max=10000):
    loss_function = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for poch in range(epoch):
        count = 0
        loss_avg = 0.0
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # learning_rate *= 0.2
        for inputs, label in data_loader:
            if count == count_max:
                break
            inputs = convert(inputs)
            # print(inputs)
            inputs = torch.FloatTensor(inputs)
            label = torch.FloatTensor(label[-1])

            inputs = autograd.Variable(inputs)
            label = autograd.Variable(label)

            model.zero_grad()

            output = model(inputs)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            loss_avg += loss
            count += 1
            if count % 1000 == 0:
                print(poch, count, loss_avg / 1000)
                loss_avg = 0.0


def lr(test_sample):
    inputs = convert(test_sample)
    # print(inputs)
    inputs = torch.FloatTensor(inputs)
    inputs = autograd.Variable(inputs, volatile=True)
    model = torch.load('lr.model')
    output = model(inputs).view(3)
    return output.data.numpy().tolist()


if __name__ == "__main__":
    args = Args()
    # data_loader = TrainDataLoader(args)
    # model = LinearRegressionModel(input_dim=30, output_dim=1)
    # train_model(model=model, learning_rate=0.001, data_loader=data_loader)
    # print("finished training")
    # model_name = 'lr.model'
    # torch.save(model, model_name)
    # print('saved model: ' + model_name)

    test_data_loader = TestDataLoader(args)
    for inputs, label in test_data_loader:
        output = lr(inputs)
        print(output)
        break

