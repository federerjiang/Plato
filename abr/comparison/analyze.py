log_file = 'result-1/log-lin.txt'


def get_ave(file):
    with open(file) as f:
        bitrate = 0
        rebuf = 0
        cv = 0
        smooth = 0
        count = 0
        for line in f:
            parse = line.split()
            bitrate += float(parse[12])
            rebuf += float(parse[4])
            cv += float(parse[6])
            smooth += float(parse[10])
            count += 1
            # if count == 12000:
            #     break
        ave_bitrate = bitrate / count
        ave_rebuf = rebuf / count
        ave_cv = cv / count
        ave_smooth = smooth / count
        reward = ave_bitrate - 33 * ave_rebuf / 1000 - 5.3 * ave_cv - ave_smooth
        print(ave_bitrate, ave_rebuf, ave_cv, ave_smooth, reward)


if __name__ == "__main__":
    log_mm = 'result-1/log-hd.txt'
    log_erp = 'result-1/log-hd-erp.txt'
    log_partial = 'result-1/log-hd-partial.txt'
    get_ave(log_mm)
    get_ave(log_erp)
    get_ave(log_partial)