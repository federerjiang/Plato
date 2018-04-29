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


def _update_tile_map(self, vp_future):
    def rotation_to_vp_tile(yaw, pitch, tile_column, tile_row, vp_length, vp_height, tile_map, tag):
        tile_length = 360 / tile_column
        tile_height = 180 / tile_row

        vp_pitch = pitch + 90
        vp_up = vp_pitch + vp_height / 2
        if vp_up > 180:
            vp_up = 179
        vp_down = vp_pitch - vp_height / 2
        if vp_down < 0:
            vp_down = 0

        vp_yaw = yaw + 180
        vp_part = 1
        vp_left = vp_yaw - vp_length / 2
        vp_right = vp_yaw + vp_length / 2
        if vp_left < 0:
            vp_part = 2
            vp_left_1 = 0
            vp_left_2 = vp_left + 360
            vp_right_1 = vp_right
            vp_right_2 = 359
        if vp_right > 360:
            vp_part = 2
            vp_right_1 = vp_right - 360
            vp_right_2 = 359
            vp_left_1 = 0
            vp_left_2 = vp_left

        def get_tiles(left, right, up, down, tag):
            col_start = math.floor(left / tile_length)
            col_end = math.floor(right / tile_length)
            row_start = math.floor(down / tile_height)
            row_end = math.floor(up / tile_height)
            count = 0
            for row in range(row_start, row_end + 1):
                for col in range(col_start, col_end + 1):
                    count += 1
                    if tile_map[row][col] != 1:  # if tile is not vp, then is set tag
                        tile_map[row][col] = tag
            return count

        tile_count = 0
        if vp_part == 1:
            tile_count = get_tiles(vp_left, vp_right, vp_up, vp_down, tag)
        if vp_part == 2:
            tile_count = get_tiles(vp_left_1, vp_right_1, vp_up, vp_down, tag)
            tile_count += get_tiles(vp_left_2, vp_right_2, vp_up, vp_down, tag)

        # print(tile_count)
        # return tile_map
    args = self.args
    tile_map = [x[:] for x in [[0] * args.tile_column] * args.tile_row]
    for rotation in vp_future:  # set vp tile tag
        pitch = rotation[1] * 180 / math.pi
        yaw = rotation[2] * 180 / math.pi
        rotation_to_vp_tile(yaw, pitch, args.tile_column, args.tile_row, args.vp_length, args.vp_height,
                                tile_map, 1)
    for rotation in vp_future:  # set ad tile tag
        pitch = rotation[1]
        yaw = rotation[2]
        rotation_to_vp_tile(yaw, pitch, args.tile_column, args.tile_row, args.ad_length, args.ad_height,
                                tile_map, 2)
    return tile_map


def get_acc(real_tile_map, pred_tile_map):
    count = [0, 0, 0]
    for row in range(6):
        for column in range(12):
            if real_tile_map[row][column] == 1:
                count[pred_tile_map[row][column]] += 1
    out_count, vp_count, ad_count = count[0], count[1], count[2]
    # get accuracy
    total_count = vp_count + ad_count + out_count
    vp_acc = vp_count / total_count
    ad_acc = ad_count / total_count
    out_acc = out_count / total_count
    return vp_acc, ad_acc, out_acc


def validate_lstm_rotation_acc(args, test_data_loader, rank, model_path, num_layers, hidden_size, length):
    def init_hidden(num_layers, hidden_size):
        hx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        cx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        hidden = (Variable(hx, volatile=True), Variable(cx, volatile=True))  # convert to Variable as late as possible
        return hidden

    model = torch.load(model_path, map_location='cpu')
    model.hidden = init_hidden(num_layers, hidden_size)

    with open(str(rank) + '.txt', 'w') as f:
        for inputs, label in test_data_loader:
            predicts = lstm_predict(args, model, inputs)
            pred_tile_map = _update_tile_map(predicts[length-30: length])
            real_tile_map = _update_tile_map(label[length-30: length])
            vp_acc, ad_acc, out_acc = get_acc(real_tile_map, pred_tile_map)
            f.write(str(vp_acc) + ' ' + str(ad_acc) + ' ' + str(out_acc) + '\n')
            print(rank, vp_acc, ad_acc, out_acc)


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
    with open(str(args.test_label_length) + 'lstm-128-1-tile-acc.txt', 'w') as outfile:
        for fname in file_names:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            os.remove(fname)


def validate_other_rotation_acc(args, model, test_data_loader, rank, length):

    with open(str(rank) + '.txt', 'w') as f:
        for inputs, label in test_data_loader:
            predicts = other_predict(args, model, inputs)
            pred_tile_map = _update_tile_map(predicts[length-30: length])
            real_tile_map = _update_tile_map(label[length-30: length])
            vp_acc, ad_acc, out_acc = get_acc(real_tile_map, pred_tile_map)
            f.write(str(vp_acc) + ' ' + str(ad_acc) + ' ' + str(out_acc) + '\n')
            print(rank, vp_acc, ad_acc, out_acc)


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
    with open(str(args.test_label_length) + '-' + name + '-tile-acc.txt', 'w') as outfile:
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

    for length in [30, 60, 90]:
        args = Args(length)
        main_validate_other(args, average, 'average')
    for length in [30, 60, 90]:
        args = Args(length)
        main_validate_other(args, lr_cal, 'lr_cal')
    for length in [30, 60, 90]:
        args = Args(length)
        main_validate_lstm(args, model_path, hidden_size, num_layers)





