import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')


def get_count_arr(file, interval):
    def _str_to_float(str_list):
        float_list = []
        for string in str_list:
            float_list.append(float(string))
        return float_list

    rolls = []
    pitchs = []
    yaws = []
    with open(file, 'r') as f:
        for line in f:
            errors = _str_to_float(line.split())
            rolls.append(errors[0])
            pitchs.append(errors[1])
            yaws.append(errors[2])

    roll_count = np.zeros(int(180/interval))
    pitch_count = np.zeros(int(180/interval))
    yaw_count = np.zeros(int(180/interval))
    for index in range(len(rolls)):
        roll_count[int(np.floor(rolls[index]/interval))] += 1
        pitch_count[int(np.floor(pitchs[index]/interval))] += 1
        yaw_count[int(np.floor(yaws[index]/interval))] += 1

    X = np.arange(0, 180, interval)
    # print(len(X), len(roll_count))
    roll_count /= roll_count.sum()
    pitch_count /= pitch_count.sum()
    yaw_count /= yaw_count.sum()
    # print(roll_count)
    roll_cdf = np.cumsum(roll_count)
    pitch_cdf = np.cumsum(pitch_count)
    yaw_cdf = np.cumsum(yaw_count)

    # plt.plot(X, pitch_count, 'g+')
    # plt.plot(X, pitch_cdf, 'r-')
    # plt.plot(X, yaw_cdf, 'g*')
    # plt.plot(X, roll_cdf, 'y>')
    # plt.title(file)
    # plt.show()
    # plt.savefig('destination_path.eps', format='eps', dpi=1000)

    return roll_cdf, pitch_cdf, yaw_cdf


def plot_yaw_cdf(averages, lrs, lstms, interval, time):
    average_roll, average_pitch, average_yaw = get_count_arr(averages[time], interval)
    lr_roll, lr_pitch, lr_yaw = get_count_arr(lrs[time], interval)
    lstm_roll, lstm_pitch, lstm_yaw = get_count_arr(lstms[time], interval)
    X = np.arange(0, 180, interval)

    rcParams.update({'figure.autolayout': True})
    rcParams['lines.linewidth'] = 2
    params = {'legend.fontsize': 20, 'legend.handlelength': 1.5}
    plt.rcParams.update(params)
    fig = plt.figure()

    line_average, = plt.plot(X, average_yaw, 'go:', label='average')
    line_lr, = plt.plot(X, lr_yaw, 'y-', label='lr')
    line_lstm, = plt.plot(X, lstm_yaw, 'r*-', label='lstm')
    plt.legend(handles=[line_lstm, line_lr, line_average])
    plt.title('error (yaw degree) CDF of time: ' + str(time+1) + 's', fontsize=20)
    plt.xlabel('degree', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.grid()
    # plt.show()
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    fig.savefig('yaw-' + str(time) + '.eps', format='eps', dpi=1000)


def plot_pitch_cdf(averages, lrs, lstms, interval, time):
    average_roll, average_pitch, average_yaw = get_count_arr(averages[time], interval)
    lr_roll, lr_pitch, lr_yaw = get_count_arr(lrs[time], interval)
    lstm_roll, lstm_pitch, lstm_yaw = get_count_arr(lstms[time], interval)
    X = np.arange(0, 180, interval)

    rcParams.update({'figure.autolayout': True})
    rcParams['lines.linewidth'] = 2
    params = {'legend.fontsize': 20, 'legend.handlelength': 1.5}
    plt.rcParams.update(params)
    fig = plt.figure()

    line_average, = plt.plot(X, average_pitch, 'go:', label='average')
    line_lr, = plt.plot(X, lr_pitch, 'y-', label='lr')
    line_lstm, = plt.plot(X, lstm_pitch, 'r*-', label='lstm')
    plt.legend(handles=[line_lstm, line_lr, line_average])
    plt.title('error (pitch degree) CDF of time: ' + str(time+1) + 's', fontsize=20)
    plt.xlabel('degree', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.grid()
    # plt.show()
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    fig.savefig('pitch-' + str(time) + '.eps', format='eps', dpi=1000)


def plot_roll_cdf(averages, lrs, lstms, interval, time):
    average_roll, average_pitch, average_yaw = get_count_arr(averages[time], interval)
    lr_roll, lr_pitch, lr_yaw = get_count_arr(lrs[time], interval)
    lstm_roll, lstm_pitch, lstm_yaw = get_count_arr(lstms[time], interval)
    X = np.arange(0, 180, interval)

    rcParams.update({'figure.autolayout': True})
    rcParams['lines.linewidth'] = 2
    params = {'legend.fontsize': 20, 'legend.handlelength': 1.5}
    plt.rcParams.update(params)
    fig = plt.figure()

    line_average, = plt.plot(X, average_roll, 'go:', label='average')
    line_lr, = plt.plot(X, lr_roll, 'y-', label='lr')
    line_lstm, = plt.plot(X, lstm_roll, 'r*-', label='lstm')
    plt.legend(handles=[line_lstm, line_lr, line_average])
    plt.title('error (roll degree) CDF of time: ' + str(time+1) + 's', fontsize=20)
    plt.xlabel('degree', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.grid()
    # plt.show()
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    fig.savefig('roll-' + str(time) + '.eps', format='eps', dpi=1000)


if __name__ == "__main__":
    averages = ['30-average-error.txt', '60-average-error.txt', '90-average-error.txt']
    lrs = ['30-lr_cal-error.txt', '60-lr_cal-error.txt', '90-lr_cal-error.txt']
    lstms = ['30lstm-128-1-error.txt', '60lstm-128-1-error.txt', '90lstm-128-1-error.txt']
    lstm256 = ['30lstm-256-1-error.txt', '60lstm-256-1-error.txt', '90lstm-256-1-error.txt']
    # get_count_arr('30-average-error.txt', 5)
    # plot_yaw_cdf(averages, lrs, lstms, 20, 0)
    # plot_yaw_cdf(averages, lrs, lstms, 20, 1)
    # plot_yaw_cdf(averages, lrs, lstms, 20, 2)
    # plot_pitch_cdf(averages, lrs, lstms, 20, 0)
    # plot_pitch_cdf(averages, lrs, lstms, 20, 1)
    # plot_pitch_cdf(averages, lrs, lstms, 20, 2)
    # plot_roll_cdf(averages, lrs, lstms, 20, 0)
    # plot_roll_cdf(averages, lrs, lstms, 20, 1)
    plot_roll_cdf(averages, lrs, lstms, 20, 2)


