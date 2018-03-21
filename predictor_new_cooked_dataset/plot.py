import numpy as np
import matplotlib.pyplot as plt


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


def plot_cdf(averages, lrs, lstms, interval):
    average_roll, average_pitch, average_yaw = get_count_arr(averages[2], interval)
    lr_roll, lr_pitch, lr_yaw = get_count_arr(lrs[2], interval)
    lstm_roll, lstm_pitch, lstm_yaw = get_count_arr(lstms[2], interval)
    X = np.arange(0, 180, interval)

    plt.plot(X, average_yaw, 'go')
    plt.plot(X, lr_yaw, 'y>')
    plt.plot(X, lstm_yaw, 'r*')
    plt.title('Prediction error (degree) CDF')
    plt.xlabel('degree')
    plt.ylabel('accuracy')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    averages = ['30-average-error.txt', '60-average-error.txt', '90-average-error.txt']
    lrs = ['30-lr_cal-error.txt', '60-lr_cal-error.txt', '90-lr_cal-error.txt']
    lstms = ['30lstm-128-1-error.txt', '60lstm-128-1-error.txt', '90lstm-128-1-error.txt']
    # get_count_arr('30-average-error.txt', 5)
    plot_cdf(averages, lrs, lstms, 5)
