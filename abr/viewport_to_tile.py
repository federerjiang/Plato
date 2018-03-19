import os
import fnmatch
import math


viewport_trace = '../datasets/viewport_trace/'
fr_dataset = viewport_trace + 'fr-dataset/'


def get_size(file):
    size_byte = os.path.getsize(file)
    size_m = size_byte / 1024 / 1024
    return size_m


def str_to_float(str_list):
    float_list = []
    for string in str_list:
        float_list.append(float(string))
    return float_list


def unit_to_rotation(unit):
    q0 = unit[0]
    q1 = unit[1]
    q2 = unit[2]
    q3 = unit[3]

    roll = math.atan2(2.0 * (q3 * q2 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    pitch = math.asin(2.0 * (q2 * q0 - q3 * q1))
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


def print_tile(tile_map, length, height):
    for row in range(height):
        for col in range(length):
            print(tile_map[row][col], end=" ")
        print('\n', end="")


def test_rotation_tile(unit_trace):
    for sample_trace in vp_trace:
        # sample = 200
        # sample_trace = sample_trace[sample]
        sample_rotation = unit_to_rotation(sample_trace[2:6])
        pitch = sample_rotation[1]
        yaw = sample_rotation[2]
        # tile_map = rotation_to_vp_tile(yaw, pitch, 12, 6, 110, 90)
        # print(tile_map)
        tile_map = rotation_to_tile(yaw, pitch, 12, 6, 110, 90, 120, 100)
        print_tile(tile_map, 12, 6)


if __name__ == '__main__':
    # count = 0
    size = 0.0
    for root, dirnames, filenames in os.walk(fr_dataset):
        for filename in fnmatch.filter(filenames, '*.txt'):
            if filename == 'formAnswers.txt' or filename == 'testInfo.txt':
                # print()
                continue
            # count += 1
            # print(root, filename)
            sample_file = os.path.join(root, filename)

    vp_trace = []
    with open(sample_file, 'r') as f:
        for line in f:
            if len(line) > 10:
                parse = line.split()
                vp_trace.append(str_to_float(parse))
    # print(len(vp_trace))
        # size += get_size(os.path.join(root, filename))
    # print(count)
    # print(size)
    test_rotation_tile(vp_trace)

