def avg_prediction(train_data):
    # loss_function = nn.MSELoss()
    loss_sum = 0.0
    for i in range(11000, 14000):
        x = 0.0
        y = 0.0
        z = 0.0
        w = 0.0
        for j in range(30):
            x += train_data[i+j][0]
            y += train_data[i+j][1]
            z += train_data[i+j][2]
            w += train_data[i+j][3]
        output = [x/30, y/30, z/30, w/30]
        label = train_data[i+30]
        loss = 0.0
        for k in range(4):
            loss += (output[k] - label[k]) * (output[k] - label[k])
        loss_sum += loss / 4
    return loss_sum / 3000