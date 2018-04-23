from ewma import EwmaBandwidthEstimator
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

BW_TRACE = '../datasets/bw_trace/'
# BW_FILE = BW_TRACE + 'sim_belgium/report_bus_0004.log'
BW_FILE = BW_TRACE + 'sim_belgium/report_car_0002.log'

def get_bw(log_file=BW_FILE):
    time = []
    throughput_all = []
    with open(log_file) as f:
        for line in f:
            parse = line.split()
            time.append(float(parse[0]))
            throughput_all.append(float(parse[1]))
    estimator = EwmaBandwidthEstimator()
    throughput_estimation = []
    for throughput in throughput_all:
        estimator.sample(1, throughput)
        estimation = estimator.get_estimate()
        print(estimation)
        throughput_estimation.append(estimation)
    return np.array(time), np.array(throughput_all), np.array(throughput_estimation)


def plot(x, y1, y2):
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.xlabel('Time (second)')
    plt.ylabel('Throughput Mbits/sec')
    plt.title('ewma estimation bus-0004')
    plt.show()


if __name__ == "__main__":
    x, y1, y2 = get_bw()
    plot(x, y1, y2)






