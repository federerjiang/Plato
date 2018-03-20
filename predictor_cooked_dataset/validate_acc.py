from average import average
from lr import lr
from cuda_lstm import LSTMPredict
from data_loader import TestDataLoader
from data_loader import TEST_LABEL_LENGTH

import torch
from torch.autograd import Variable
import math


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
                if tile_map[row][col] == 0:
                    tile_map[row][col] = tag
        return count

    tile_count = 0
    if vp_part == 1:
        tile_count = get_tiles(vp_left, vp_right, vp_up, vp_down, tag)
    if vp_part == 2:
        tile_count = get_tiles(vp_left_1, vp_right_1, vp_up, vp_down, tag)
        tile_count += get_tiles(vp_left_2, vp_right_2, vp_up, vp_down, tag)

    print(tile_count)
    return tile_map


def rotation_to_tile(yaw, pitch, tile_column, tile_row, vp_length, vp_height, ad_length, ad_height):
    tile_map = [x[:] for x in [[0] * tile_column] * tile_row]
    tile_map = rotation_to_vp_tile(yaw, pitch, 12, 6, vp_length, vp_height, tile_map, 1)
    tile_map = rotation_to_vp_tile(yaw, pitch, 12, 6, ad_length, ad_height, tile_map, 2)
    return tile_map


def lstm_predict(model, inputs, length=TEST_LABEL_LENGTH):
    outputs = []
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


def other_predict(model, inputs, length=TEST_LABEL_LENGTH):
    outputs = []
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


def validate_lstm_rotation_acc(test_data_loader, model_path, num_layers, hidden_size):
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
    with open(str(TEST_LABEL_LENGTH) + 'lstm-128-1-loss.txt', 'w') as f:
        for inputs, label in test_data_loader:
            predicts = lstm_predict(model, inputs)
            predict_rolls, predict_pitchs, predict_yaws, label_rolls, label_pitchs, label_yaws = get_rotations(predicts,
                                                                                                               label)
            roll_loss = get_loss(loss_function, predict_rolls, label_rolls)
            pitch_loss = get_loss(loss_function, predict_pitchs, label_pitchs)
            yaw_loss = get_loss(loss_function, predict_yaws, label_yaws)

            f.write(str(roll_loss) + ' ' + str(pitch_loss) + ' ' + str(yaw_loss) + '\n')
            print(roll_loss, pitch_loss, yaw_loss)
        # print(loss)
        # if loss <= 10:
        #     count_acc += 1
            count += 1
        # loss_sum += loss
            if count == 100000:
                break
    print(count_acc)
    print(count_acc / count)
    print(loss_sum / count)
    return loss_sum / count


def validate_lstm_tile_acc(test_data_loader, model_path, num_layers, hidden_size):
    pass


def validate_other_rotation_acc(model, test_data_loader):

    loss_function = torch.nn.MSELoss()
    loss_sum = 0.0
    count = 0
    count_acc = 0
    with open(str(TEST_LABEL_LENGTH) + 'lr-loss.txt', 'w') as f:
        for inputs, label in test_data_loader:
            predicts = other_predict(model, inputs)
            predict_rolls, predict_pitchs, predict_yaws, label_rolls, label_pitchs, label_yaws = get_rotations(predicts,
                                                                                                               label)
            roll_loss = get_loss(loss_function, predict_rolls, label_rolls)
            pitch_loss = get_loss(loss_function, predict_pitchs, label_pitchs)
            yaw_loss = get_loss(loss_function, predict_yaws, label_yaws)

            f.write(str(roll_loss) + ' ' + str(pitch_loss) + ' ' + str(yaw_loss) + '\n')
            print(roll_loss, pitch_loss, yaw_loss)
        # print(loss)
        # if loss <= 10:
        #     count_acc += 1
            count += 1
            # loss_sum += loss
            if count == 100000:
                break
    # print(count_acc)
    # print(count_acc / count)
    # print(loss_sum / count)
    return loss_sum / count


if __name__ == "__main__":
    test_data_loader = TestDataLoader()
    validate_lstm_rotation_acc(test_data_loader, 'adam-lstm-128-1.model', 1, 128)
    # validate_other_rotation_acc(average, test_data_loader)
    # validate_other_rotation_acc(lr, test_data_loader)



