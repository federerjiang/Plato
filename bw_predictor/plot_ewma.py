from ewma import EwmaBandwidthEstimator
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')

BW_TRACE = '../datasets/bw_trace/'
BW_FILE = BW_TRACE + 'sim_belgium/report_bus_0007.log'
# BW_FILE = BW_TRACE + 'sim_belgium/report_bicycle_0001.log'
# BW_FILE = BW_TRACE + 'sim_belgium/report_car_0002.log'

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
    rcParams.update({'figure.autolayout': True})
    rcParams['lines.linewidth'] = 2
    params = {'legend.fontsize': 20, 'legend.handlelength': 1.5}
    plt.rcParams.update(params)
    fig = plt.figure()

    real, = plt.plot(x, y1, 'black', label='Real bandwidth')
    estimate, = plt.plot(x, y2, 'r*-', label='EWMA estimation')
    plt.legend(handles=[real, estimate])
    plt.xlabel('Time (second)')
    plt.ylabel('Throughput Mbits/sec')
    plt.title('ewma estimation bus-0004')
    plt.grid()
    # fig.savefig('bicycle-0001' + '.eps', format='eps', dpi=1000)
    fig.savefig('bus-0007' + '.eps', format='eps', dpi=1000)
    # plt.show()


if __name__ == "__main__":
    x, y1, y2 = get_bw()
    plot(x, y1, y2)






