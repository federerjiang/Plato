import os
import fnmatch

VIEWPORT_TRACE = '../datasets/viewport_trace/'
TRAIN_DATASET = VIEWPORT_TRACE + 'new_cooked_train_dataset/'
TEST_DATASET = VIEWPORT_TRACE + 'new_cooked_test_dataset/'


def load_viewport_unit(trace_folder=TRAIN_DATASET):
    all_vp_time = []
    all_vp_unit = []
    for root, dirnames, filenames in os.walk(trace_folder):
        for filename in fnmatch.filter(filenames, '*.txt'):
            if filename == 'formAnswers.txt' or filename == 'testInfo.txt':
                continue
            trace_file = os.path.join(root, filename)
            vp_time = []
            vp_unit = []
            with open(trace_file, 'r') as f:
                for line in f:
                    if len(line) > 10:
                        parse = line.split()
                        vp_time.append(float(parse[0]))
                        vp_unit.append(_str_to_float(parse[2:5]))
            all_vp_time.append(vp_time)
            all_vp_unit.append(vp_unit)

    return all_vp_time, all_vp_unit


def _str_to_float(str_list):
    float_list = []
    for string in str_list:
        float_list.append(float(string))
    return float_list

if __name__ == '__main__':
    all_vp_time, all_vp_unit = load_viewport_unit()
    print(len(all_vp_unit))
