from average import average
from lr import lr
from cuda_lstm import LSTMPredict
from data_loader import TestDataLoader

import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
import math
from args import Args

import os


def lstm_predict(model, inputs, args):
    outputs = []
    length = args.test_label_length
    inputs = torch.FloatTensor(inputs).view(1, 30, 3)
    inputs = Variable(inputs, volatile=True)
    for _ in range(length):
        output = model(inputs)
        outputs.append(output[-1].data.numpy().tolist())
        t = Variable(torch.randn(1, 30, 3), volatile=True)
        t[:, 0:29, :] = inputs[:, 1:30, :]
        t[:, 29, :] = output.view(1, 30, 3)[:, 29, :]
        inputs = t
    return outputs


def other_predict(model, inputs, args):
    outputs = []
    length = args.test_label_length
    for _ in range(length):
        output = model(inputs)
        outputs.append(output)
        inputs.pop(0)
        inputs.append(output)
    # print(len(outputs))
    return outputs


def get_rotations(predicts, label):
    predict_rolls = []
    predict_pitchs = []
    predict_yaws = []
    label_rolls = []
    label_pitchs = []
    label_yaws = []

    for sample in predicts:
        predict_rolls.append(sample[0])
        predict_pitchs.append(sample[1])
        predict_yaws.append(sample[2])
    for sample in label:
        label_rolls.append(sample[0])
        label_pitchs.append((sample[1]))
        label_yaws.append(sample[2])
    return predict_rolls, predict_pitchs, predict_yaws, label_rolls, label_pitchs, label_yaws


def get_loss(loss_function, predicts, label):
    predicts = torch.FloatTensor(predicts)
    predicts = Variable(predicts, volatile=True)
    label = torch.FloatTensor(label)
    label = Variable(label, volatile=True)
    loss = loss_function(predicts, label)
    loss = math.sqrt(loss) * 180
    return loss


def validate_lstm_rotation_acc(args, test_data_loader, rank, model_path, num_layers, hidden_size):
    def init_hidden(num_layers, hidden_size):
        hx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        cx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        hidden = (Variable(hx, volatile=True), Variable(cx, volatile=True))  # convert to Variable as late as possible
        return hidden

    model = torch.load(model_path, map_location='cpu')
    model.hidden = init_hidden(num_layers, hidden_size)

    loss_function = torch.nn.MSELoss()
    loss_sum = 0.0
    count = 0
    count_acc = 0
    with open(str(rank) + '-' + str(args.test_label_length) + '(30-60)lstm-128-1-loss.txt', 'w') as f:
        for inputs, label in test_data_loader:
            predicts = lstm_predict(model, inputs)
            predict_rolls, predict_pitchs, predict_yaws, label_rolls, label_pitchs, label_yaws = \
                get_rotations(predicts[30:60], label[30:60])
            roll_loss = get_loss(loss_function, predict_rolls, label_rolls)
            pitch_loss = get_loss(loss_function, predict_pitchs, label_pitchs)
            yaw_loss = get_loss(loss_function, predict_yaws, label_yaws)

            f.write(str(roll_loss) + ' ' + str(pitch_loss) + ' ' + str(yaw_loss) + '\n')
            print(roll_loss, pitch_loss, yaw_loss)
            count += 1
            if count == 100000:
                break
    print(count_acc)
    print(count_acc / count)
    print(loss_sum / count)
    return loss_sum / count


def validate_other_rotation_acc(args, model, test_data_loader):

    loss_function = torch.nn.MSELoss()
    loss_sum = 0.0
    count = 0
    with open(str(args.test_label_length) + '(30-60)lr-loss.txt', 'w') as f:
        for inputs, label in test_data_loader:
            predicts = other_predict(model, inputs)
            predict_rolls, predict_pitchs, predict_yaws, label_rolls, label_pitchs, label_yaws = \
                get_rotations(predicts[30:60], label[30:60])
            roll_loss = get_loss(loss_function, predict_rolls, label_rolls)
            pitch_loss = get_loss(loss_function, predict_pitchs, label_pitchs)
            yaw_loss = get_loss(loss_function, predict_yaws, label_yaws)

            f.write(str(roll_loss) + ' ' + str(pitch_loss) + ' ' + str(yaw_loss) + '\n')
            print(roll_loss, pitch_loss, yaw_loss)
            count += 1
            if count == 100000:
                break
    return loss_sum / count


if __name__ == "__main__":
    args = Args()
    test_data_loader = TestDataLoader(args)  # total 114857 samples for test

    # validate_lstm_rotation_acc(test_data_loader, 'adam-lstm-128-1.model', 1, 128)
    # validate_other_rotation_acc(average, test_data_loader)
    # validate_other_rotation_acc(lr, test_data_loader)
    new_cooked_test_dataset = '../datasets/viewport_trace/new_cooked_test_dataset/'
    sub_paths = []
    for uid in os.listdir(new_cooked_test_dataset):
        sub_paths.append(os.path.join(new_cooked_test_dataset, uid))
    # print(os.listdir(new_cooked_test_dataset))
    count = 0
    for sub_path in sub_paths:
        test_data_loader = TestDataLoader(args, trace_folder=sub_path)
        count = 0
        for i in test_data_loader:
            count += 1
        print(count)


