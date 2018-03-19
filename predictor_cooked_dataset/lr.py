import numpy as np


def lr(test_sample):
    x = []
    y = []
    z = []
    w = []

    for unit in test_sample:
        x.append(unit[0])
        y.append(unit[1])
        z.append(unit[2])
        w.append(unit[3])

    output = np.zeros(4)
    sample_length = len(test_sample)
    m, c = linear_regression(x)
    output[0] = m * sample_length + c
    m, c = linear_regression(y)
    output[1] = m * sample_length + c
    m, c = linear_regression(z)
    output[2] = m * sample_length + c
    m, c = linear_regression(w)
    output[3] = m * sample_length + c

    return output


def linear_regression(orientations):
    x = np.arange(len(orientations))
    y = np.array(orientations)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c




# def data_loader(filename):
#     train_data = []
#     with open(filename) as f:
#         for line in f:
#             cordinates = line.split(",")
#             if len(cordinates) == 4:
#                 for i in range(4):
#                     cordinates[i] = float(cordinates[i])
#             train_data.append(cordinates)
#     return train_data

if __name__ == "__main__":
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1])
    print(linear_regression(y))
    # train_data = data_loader("../datasets/pre_train.csv")
    # loss_sum = 0.0
    # for i in range(11000, 14000):
    #     x = []
    #     y = []
    #     z = []
    #     w = []
    #     for j in range(30):
    #         x.append(train_data[i+j][0])
    #         y.append(train_data[i+j][1])
    #         z.append(train_data[i+j][2])
    #         w.append(train_data[i+j][3])
    #     label = train_data[i+30]
    #     m,c = linear_regression(x)
    #     output = []
    #     output.append(m*30 + c)
    #     m,c = linear_regression(y)
    #     output.append(m*30 + c)
    #     m,c = linear_regression(z)
    #     output.append(m*30 + c)
    #     m,c = linear_regression(w)
    #     output.append(m*30 + c)
    #     loss = 0.0
    #     for k in range(4):
    #         loss += (output[k] - label[k]) * (output[k] - label[k])
    #     loss_sum += loss / 4
    # print(loss_sum / 3000)