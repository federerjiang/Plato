from average import average
from lr import lr
from lr_cal import lr_cal
from cuda_lstm import LSTMPredict
from lr import LinearRegressionModel
from data_loader import TestDataLoader

import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
import math
from args import Args

import os


def lstm_predict(args, model, inputs):
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


def other_predict(args, model, inputs):
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


def get_loss_360(predicts, label):
    sum_error = 0
    length = len(predicts)
    for i in range(length):
        err = abs(predicts[i] - label[i])
        if err > 1:  # it's in a sphere, the largest error is one (half sphere)
            err = 2 - err
        sum_error += err
    loss = (sum_error / length) * 180
    return loss


def validate_lstm_rotation_acc(args, test_data_loader, rank, model_path, num_layers, hidden_size, length):
    def init_hidden(num_layers, hidden_size):
        hx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        cx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        hidden = (Variable(hx, volatile=True), Variable(cx, volatile=True))  # convert to Variable as late as possible
        return hidden

    model = torch.load(model_path, map_location='cpu')
    model.hidden = init_hidden(num_layers, hidden_size)

    loss_function = torch.nn.MSELoss()
    with open(str(rank) + '.txt', 'w') as f:
        for inputs, label in test_data_loader:
            predicts = lstm_predict(args, model, inputs)
            predict_rolls, predict_pitchs, predict_yaws, label_rolls, label_pitchs, label_yaws = \
                get_rotations(predicts[length-30: length], label[length-30: length])
            # roll_loss = get_loss_360(loss_function, predict_rolls, label_rolls)
            roll_loss = get_loss_360(predict_rolls, label_rolls)
            pitch_loss = get_loss(loss_function, predict_pitchs, label_pitchs)
            yaw_loss = get_loss_360(predict_yaws, label_yaws)
            # yaw_loss = get_loss_360(loss_function, predict_yaws, label_yaws)

            f.write(str(roll_loss) + ' ' + str(pitch_loss) + ' ' + str(yaw_loss) + '\n')
            print(roll_loss, pitch_loss, yaw_loss)


def main_validate_lstm(args, model_path, hidden_size, num_layers):
    new_cooked_test_dataset = '../datasets/viewport_trace/new_cooked_test_dataset/'
    sub_paths = []
    for uid in os.listdir(new_cooked_test_dataset):
        sub_paths.append(os.path.join(new_cooked_test_dataset, uid))
    test_data_loaders = []
    for sub_path in sub_paths:
        test_data_loader = TestDataLoader(args, trace_folder=sub_path)
        test_data_loaders.append(test_data_loader)
    processes = []
    for rank in range(len(test_data_loaders)):
        p = mp.Process(target=validate_lstm_rotation_acc, args=(args, test_data_loaders[rank], rank,
                                                                model_path, num_layers, hidden_size, args.test_label_length))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    file_names = []
    for rank in range(len(sub_paths)):
        file_names.append(str(rank) + '.txt')
    with open(str(args.test_label_length) + 'lstm-128-1-error.txt', 'w') as outfile:
        for fname in file_names:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            os.remove(fname)


def validate_other_rotation_acc(args, model, test_data_loader, rank, length):

    loss_function = torch.nn.MSELoss()
    loss_sum = 0.0
    count = 0
    with open(str(rank) + '.txt', 'w') as f:
        for inputs, label in test_data_loader:
            predicts = other_predict(args, model, inputs)
            predict_rolls, predict_pitchs, predict_yaws, label_rolls, label_pitchs, label_yaws = \
                get_rotations(predicts[length-30: length], label[length-30: length])
            roll_loss = get_loss_360(predict_rolls, label_rolls)
            pitch_loss = get_loss(loss_function, predict_pitchs, label_pitchs)
            yaw_loss = get_loss_360(predict_yaws, label_yaws)

            f.write(str(roll_loss) + ' ' + str(pitch_loss) + ' ' + str(yaw_loss) + '\n')
            print(rank, roll_loss, pitch_loss, yaw_loss)


def main_validate_other(args, model, name):
    new_cooked_test_dataset = '../datasets/viewport_trace/new_cooked_test_dataset/'
    sub_paths = []
    for uid in os.listdir(new_cooked_test_dataset):
        sub_paths.append(os.path.join(new_cooked_test_dataset, uid))
    test_data_loaders = []
    for sub_path in sub_paths:
        test_data_loader = TestDataLoader(args, trace_folder=sub_path)
        test_data_loaders.append(test_data_loader)
    processes = []
    for rank in range(len(test_data_loaders)):
        p = mp.Process(target=validate_other_rotation_acc, args=(args, model, test_data_loaders[rank], rank, args.test_label_length))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    file_names = []
    for rank in range(len(sub_paths)):
        file_names.append(str(rank) + '.txt')
    with open(str(args.test_label_length) + '-' + name + '-error.txt', 'w') as outfile:
        for fname in file_names:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            os.remove(fname)


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(torch.get_num_threads())
    model_path = 'adam-lstm-128-1.model'
    hidden_size = 128
    num_layers = 1
    # test_data_loader = TestDataLoader(args)  # total 114857 samples for test

    # validate_lstm_rotation_acc(test_data_loader, 'adam-lstm-128-1.model', 1, 128)
    # validate_other_rotation_acc(args, average, test_data_loader)
    # validate_other_rotation_acc(args, lr, test_data_loader)
    # for length in [30, 60, 90]:
    #     args = Args(length)
    #     main_validate_other(args, lr_cal, 'lr_cal')
    for length in [30, 60, 90]:
        args = Args(length)
        main_validate_other(args, average, 'average')
    # for length in [30, 60, 90]:
    #     args = Args(length)
    #     main_validate_lstm(args, model_path, hidden_size, num_layers)





