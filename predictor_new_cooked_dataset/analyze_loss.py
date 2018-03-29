

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
    count = 0
    with open(file_path) as f:
        for line in f:
            parse = line.split()
            if float(parse[0]) < max:
                roll += 1
            if float(parse[1]) < max:
                pitch += 1
            if float(parse[2]) < max:
                yaw += 1
            count += 1
    return roll/count, pitch/count, yaw/count


def try_error(error1, error2, error3):
    print('(0 - 30) frames: 1st second')
    result = get_thre_count('30lstm-128-1-error.txt', error1)
    print(error1, result)
    result = get_thre_count('30-lr_cal-error.txt', error1)
    print(error1, result)
    result = get_thre_count('30-average-error.txt', error1)
    print(error1, result)

    print('(30 - 60) frames: 2nd second')
    result = get_thre_count('60lstm-128-1-error.txt', error2)
    print(error2, result)
    result = get_thre_count('60-lr_cal-error.txt', error2)
    print(error2, result)
    result = get_thre_count('60-average-error.txt', error2)
    print(error2, result)

    print('(60 - 90) frames: 3rd second')
    result = get_thre_count('90lstm-128-1-error.txt', error3)
    print(error3, result)
    result = get_thre_count('90-lr_cal-error.txt', error3)
    print(error3, result)
    result = get_thre_count('90-average-error.txt', error3)
    print(error3, result)


def get_average_error():
    print('0 - 30 frames: 1st second')
    result = get_average_rotation('30lstm-128-1-error.txt')
    print(result)
    result = get_average_rotation('30-lr_cal-error.txt')
    print(result)
    result = get_average_rotation('30-average-error.txt')
    print(result)

    print('30 - 60 frames: 2nd second')
    result = get_average_rotation('60lstm-128-1-error.txt')
    print(result)
    result = get_average_rotation('60-lr_cal-error.txt')
    print(result)
    result = get_average_rotation('60-average-error.txt')
    print(result)

    print('60 - 90 frames: 3rd second')
    result = get_average_rotation('90lstm-128-1-error.txt')
    print(result)
    result = get_average_rotation('90-lr_cal-error.txt')
    print(result)
    result = get_average_rotation('90-average-error.txt')
    print(result)


if __name__ == "__main__":

    try_error(30, 30, 50)
    # get_average_error()
