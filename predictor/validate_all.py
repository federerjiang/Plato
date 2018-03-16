import torch
from predictor.data_loader import LABEL_SAMPLE_LENGTH  # in mac

LSTM_MODEL_PATH = 'lstm-128-1.model'


def lstm_predict(test_data_loader):
    pass

def validate(model, data_loader):
    loss_function = nn.MSELoss()
    loss_sum = 0.0
    for inputs, label in data_loader:
        # inputs = train_data[i: i+30]
        inputs = torch.FloatTensor(inputs).view(30, 1, -1)
        inputs = autograd.Variable(inputs)
        # label = torch.FloatTensor(train_data[i+30])
        label = autograd.Variable(label)
        output = model(inputs)
        loss = loss_function(output[-1], label)
        loss_sum += loss
    return loss_sum / 3000