

def get_average(path):
    reward_sum = 0
    count = 0
    with open(path) as f:
        for line in f:
            parse = line.split()
            reward = float(parse[-3])
            # print(reward)
            reward_sum += reward
            count += 1
    return reward_sum / count


if __name__ == "__main__":
    nums = [20000, 40000, 45000, 55000]
    for num in nums:
        file_path = 'result-1/log-' + str(num) + '.txt'  # 20000, 40000, 45000, 55000
        ave = get_average(file_path)
        print('num: ', num)
        print('average reward: ', ave)
    comparison_path = '../comparison/result-1/log-1.txt'
    ave = get_average(comparison_path)
    print('average reward: ', ave)
