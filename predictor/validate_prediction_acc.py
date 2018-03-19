from validate_prediction_loss import lstm_predict
from validate_prediction_loss import other_predict
from average import average
from lr import lr
from cuda_lstm import LSTMPredict
from data_loader import TestDataLoader

import torch
from torch.autograd import Variable
import math


def unit_to_rotation(unit):
    q0 = unit[0]
    q1 = unit[1]
    q2 = unit[2]
    q3 = unit[3]

    roll = math.atan2(2.0 * (q3 * q2 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    val = 2.0 * (q2 * q0 - q3 * q1)
    val = max(-1, min(1, val))
    pitch = math.asin(val)
    yaw = math.atan2(2.0 * (q3 * q0 + q1 * q2), -1.0 + 2.0 * (q0 * q0 + q1 * q1))
    return roll*180/math.pi, pitch*180/math.pi, yaw*180/math.pi


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


def validate_lstm_rotation_acc(test_data_loader, model_path, num_layers, hidden_size):
    def init_hidden(num_layers, hidden_size):
        hx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        cx = torch.nn.init.xavier_normal(torch.randn(num_layers, 1, hidden_size))
        hidden = (Variable(hx, volatile=True), Variable(cx, volatile=True))  # convert to Variable as late as possible
        return hidden

    def units_to_rotations(units):
        rotations = []
        for unit in units:
            _, pitch, yaw = unit_to_rotation(unit)
            rotation = list()
            rotation.append(pitch)
            rotation.append(yaw)
            rotations.append(rotation)
        return rotations

    model = torch.load(model_path, map_location='cpu')
    model.hidden = init_hidden(num_layers, hidden_size)

    loss_function = torch.nn.MSELoss()
    loss_sum = 0.0
    count = 0
    count_acc = 0
    for inputs, label in test_data_loader:
        predicts = lstm_predict(model, inputs)
        predict_rotations = units_to_rotations(predicts)
        label_rotations = units_to_rotations(label)
        predict_rotations = torch.FloatTensor(predict_rotations)
        predict_rotations = Variable(predict_rotations, volatile=True)
        label_rotations = torch.FloatTensor(label_rotations)
        label_rotations = Variable(label_rotations, volatile=True)
        loss = loss_function(predict_rotations, label_rotations)
        loss = math.sqrt(loss)
        print(loss)
        if loss <= 10:
            count_acc += 1
        count += 1
        loss_sum += loss
        if count == 10000:
            break
    print(count_acc)
    print(count_acc / count)
    return loss_sum / count


def validate_lstm_tile_acc(test_data_loader, model_path, num_layers, hidden_size):
    pass


def validate_other_rotation_acc(model, test_data_loader):
    def units_to_rotations(units):
        rotations = []
        for unit in units:
            _, pitch, yaw = unit_to_rotation(unit)
            rotation = list()
            rotation.append(pitch)
            rotation.append(yaw)
            rotations.append(rotation)
        return rotations

    loss_function = torch.nn.MSELoss()
    loss_sum = 0.0
    count = 0
    count_acc = 0
    for inputs, label in test_data_loader:
        predicts = other_predict(model, inputs)
        predict_rotations = units_to_rotations(predicts)
        label_rotations = units_to_rotations(label)
        predict_rotations = torch.FloatTensor(predict_rotations)
        predict_rotations = Variable(predict_rotations, volatile=True)
        label_rotations = torch.FloatTensor(label_rotations)
        label_rotations = Variable(label_rotations, volatile=True)
        loss = loss_function(predict_rotations, label_rotations)
        loss = math.sqrt(loss)
        print(loss)
        if loss <= 10:
            count_acc += 1
        count += 1
        loss_sum += loss
        if count == 10000:
            break
    print(count_acc)
    print(count_acc / count)
    return loss_sum / count


if __name__ == "__main__":
    test_data_loader = TestDataLoader()
    validate_lstm_rotation_acc(test_data_loader, 'adam-lstm-128-2.model', 2, 128)
    # validate_other_rotation_acc(average, test_data_loader)
    # validate_other_rotation_acc(lr, test_data_loader)



