

def get_average_rotation(file_path):
    roll = 0
    pitch = 0
    yaw = 0
    count = 0
    with open(file_path) as f:
        for line in f:
            parse = line.split()
            roll += float(parse[0])
            pitch += float(parse[1])
            yaw += float(parse[2])
            count += 1
    return roll/count, pitch/count, yaw/count


def get_thre_count(file_path, max):
    roll = 0
    pitch = 0
    yaw = 0
    with open(file_path) as f:
        for line in f:
            parse = line.split()
            if float(parse[0]) < max:
                roll += 1
            if float(parse[1]) < max:
                pitch += 1
            if float(parse[2]) < max:
                yaw += 1
    return roll, pitch, yaw


if __name__ == "__main__":
    result = get_average_rotation('30-average-error.txt')
    print(result)
    result = get_average_rotation('30-lr-error.txt')
    print(result)
    result = get_average_rotation('30lstm-128-1-error.txt')
    print(result)
    # result = get_average_rotation('60average-loss.txt')
    # print(result)
    # result = get_average_rotation('60lr-loss.txt')
    # print(result)
    # result = get_average_rotation('60lstm-128-1-loss.txt')
    # print(result)

    # result = get_thre_count('30average-loss.txt', 20)
    # print(result)
    # result = get_thre_count('30lr-loss.txt', 20)
    # print(result)
    # result = get_thre_count('30lstm-128-1-loss.txt', 20)
    # print(result)
    # result = get_thre_count('60average-loss.txt', 20)
    # print(result)
    # result = get_thre_count('60lr-loss.txt', 20)
    # print(result)
    # result = get_thre_count('60lstm-128-1-loss.txt', 20)
    # print(result)
    # result = get_thre_count('90average-loss.txt', 60)
    # print(result)
    # result = get_thre_count('90lr-loss.txt', 60)
    # print(result)
    # result = get_thre_count('90lstm-128-1-loss.txt', 20)
    # print(result)
