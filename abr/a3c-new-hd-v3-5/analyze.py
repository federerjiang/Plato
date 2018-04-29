log_file = 'result-1/log-20000.txt'

with open(log_file) as f:
    bitrate = 0
    rebuf = 0
    cv = 0
    smooth = 0
    count = 0
    for line in f:
        parse = line.split()
        bitrate += float(parse[8])
        rebuf += float(parse[4])
        cv += float(parse[6])
        smooth += float(parse[10])
        count += 1
        if count == 10000:
            break
    ave_bitrate = bitrate / count
    ave_rebuf = rebuf / count
    ave_cv = cv / count
    ave_smooth = smooth / count
    reward = ave_bitrate - 33 * ave_rebuf / 1000 - 5.3 * ave_cv - ave_smooth
    print(ave_bitrate, ave_rebuf, ave_cv, ave_smooth, reward)