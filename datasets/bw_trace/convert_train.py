import os
import numpy as np

BITS_IN_BYTE = 8.0
BITS_IN_MBITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0


def convert(in_path, out_path):

    files = os.listdir(in_path)
    for trace_file in files:
        with open(in_path + trace_file, 'r') as f, open(out_path + trace_file, 'w') as wf:
            time_ms = []  # time stamp
            throughput_all = []
            for line in f:
                parse = line.split()
                time_ms.append(float(parse[0]))
                throughput_all.append(float(parse[1]))

            time_ms = np.array(time_ms)
            throughput_all = np.array(throughput_all)
            throughput_all *= np.random.uniform(10, 25)
            print(trace_file)

            assert len(time_ms) == len(throughput_all)
            for i in range(len(time_ms)):
                wf.write(str(time_ms[i]) + ' ' + str(throughput_all[i]) + '\n')


if __name__ == '__main__':
    in_path = './train_traces/'
    out_path = './train_sim_traces_10-25/'
    convert(in_path, out_path)
