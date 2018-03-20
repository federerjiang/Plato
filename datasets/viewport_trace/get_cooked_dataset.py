import os
import fnmatch
import math


train_dataset = 'train-dataset/'
test_dataset = 'test-dataset/'
cooked_train_dataset = 'new_cooked_train_dataset/'
cooked_test_dataset = 'new_cooked_test_dataset/'


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
    # pitch = math.asin(2.0 * (q2 * q0 - q3 * q1))
    val = 2.0 * (q2 * q0 - q3 * q1)
    val = max(-1, min(1, val))
    pitch = math.asin(val)
    yaw = math.atan2(2.0 * (q3 * q0 + q1 * q2), -1.0 + 2.0 * (q0 * q0 + q1 * q1))
    return roll/math.pi, pitch/math.pi, yaw/math.pi


if __name__ == '__main__':
    count = 0
    for root, dirnames, filenames in os.walk(test_dataset):
        for filename in fnmatch.filter(filenames, '*.txt'):
            if filename == 'formAnswers.txt' or filename == 'testInfo.txt':
                continue
            file_path = os.path.join(root, filename)
            # print(file_path)
            uid = root.split('/')[1]
            cooked_dir = os.path.join(cooked_test_dataset, uid)
            cooked_file = os.path.join(cooked_dir, filename)
            if not os.path.isdir(cooked_dir):
                os.mkdir(cooked_dir)
            with open(file_path, 'r') as fr, open(cooked_file, 'w') as fw:
                count = -1
                for line in fr:
                    # count += 1
                    if len(line) > 10:
                        parse = line.split()
                        time = str(parse[0])
                        frame = int(parse[1])
                        if frame > count:
                            roll, pitch, yaw = unit_to_rotation(str_to_float(parse[2:6]))
                            fw.write(time + ' ' + str(frame) + ' ' + str(roll) + ' ' + str(pitch) + ' ' + str(yaw) + '\n')
                            count = frame

    print(count)
