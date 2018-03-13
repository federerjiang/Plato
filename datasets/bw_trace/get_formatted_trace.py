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
            bytes_recv = []
            time_recv = []
            for line in f:
                parse = line.split()
                time_ms.append(float(parse[1]))
                bytes_recv.append(float(parse[4]))
                time_recv.append(float(parse[5]))

            time_ms = np.array(time_ms)
            bytes_recv = np.array(bytes_recv)
            time_recv = np.array(time_recv)
            throughput_all = bytes_recv / time_recv

            print(trace_file)
            time_ms = time_ms - time_ms[0]
            time_ms /= MILLISECONDS_IN_SECONDS
            throughput_all *= BITS_IN_BYTE / BITS_IN_MBITS * MILLISECONDS_IN_SECONDS

            assert len(time_ms) == len(throughput_all)
            for i in range(len(time_ms)):
                wf.write(str(time_ms[i]) + ' ' + str(throughput_all[i]) + '\n')


if __name__ == '__main__':
    out_path = './sim_belgium/'
    in_path = './belgium/'
    convert(in_path, out_path)
