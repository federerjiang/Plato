import numpy as np


def lr(test_sample):
    x = []
    y = []
    z = []

    for unit in test_sample:
        x.append(unit[0])
        y.append(unit[1])
        z.append(unit[2])

    output = np.zeros(3)
    sample_length = len(test_sample)
    m, c = linear_regression(x)
    output[0] = m * sample_length + c
    m, c = linear_regression(y)
    output[1] = m * sample_length + c
    m, c = linear_regression(z)
    output[2] = m * sample_length + c
    return output


def linear_regression(orientations):
    x = np.arange(len(orientations))
    y = np.array(orientations)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


if __name__ == "__main__":
    x = np.array([0, 1, 2])
    y = np.array([-1, 0.2, 0.9])
    print(linear_regression(y))
