import numpy as np
import statsmodels.api as sm


def wl_regression(orientations):
    x = np.arange(len(orientations))
    y = np.array(orientations)
    x = sm.add_constant(x)
    weights = np.arange(0, 1, 0.0333)[1:]
    # weights = [x * 0.0333 for x in range(1, 31)]
    # for i in range(len(weights)):
    #     weights[i] = weights[i] * weights[i]
    wls_model = sm.WLS(y, x, weights)
    results = wls_model.fit()
    m, c = results.params
    # print(m, c)
    return m, c

# x = np.array([0, 1, 2, 3])
# y = np.array([-1, 0.2, 0.9, 2.1])
# print(wl_regression(y))
# print(np.arange(0, 1, 0.033)[1:])


def data_loader(filename):
    train_data = []
    with open(filename) as f:
        for line in f:
            cordinates = line.split(",")
            if len(cordinates) == 4:
                for i in range(4):
                    cordinates[i] = float(cordinates[i])
            train_data.append(cordinates)
    return train_data

if __name__ == "__main__":
    train_data = data_loader("../datasets/pre_train.csv")
    loss_sum = 0.0
    for i in range(11000, 14000):
        x = []
        y = []
        z = []
        w = []
        for j in range(30):
            x.append(train_data[i+j][0])
            y.append(train_data[i+j][1])
            z.append(train_data[i+j][2])
            w.append(train_data[i+j][3])
        label = train_data[i+30]
        m,c = wl_regression(x)
        output = []
        output.append(m*30 + c)
        m,c = wl_regression(y)
        output.append(m*30 + c)
        m,c = wl_regression(z)
        output.append(m*30 + c)
        m,c = wl_regression(w)
        output.append(m*30 + c)
        loss = 0.0
        for k in range(4):
            loss += (output[k] - label[k]) * (output[k] - label[k])
        loss_sum += loss / 4
    print(loss_sum / 3000)