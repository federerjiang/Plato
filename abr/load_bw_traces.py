import os

TRACE_FOLDER = '../datasets/bw_trace/sim_belgium/'


def load_trace(trace_folder=TRACE_FOLDER):
    cooked_files = os.listdir(trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'r') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names


if __name__ == '__main__':
    all_cooked_time, all_cooked_bw, _ = load_trace()
    print(all_cooked_bw)
