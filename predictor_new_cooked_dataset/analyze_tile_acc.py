

def get_average(file_path):
    vp_acc = 0
    ad_acc = 0
    out_acc = 0
    count = 0
    with open(file_path) as f:
        for line in f:
            parse = line.split()
            vp_acc += float(parse[0])
            ad_acc += float(parse[1])
            out_acc += float(parse[2])
            count += 1
    return vp_acc/count, ad_acc/count, out_acc/count


def get_average_tile_acc():
    print('0 - 30 frames: 1st second')
    vp_ave, ad_ave, out_ave = get_average('30lstm-128-1-tile-acc.txt')
    print(vp_ave, ad_ave, out_ave, vp_ave+ad_ave)
    vp_ave, ad_ave, out_ave = get_average('30-lr_cal-tile-acc.txt')
    print(vp_ave, ad_ave, out_ave)
    vp_ave, ad_ave, out_ave = get_average('30-average-tile-acc.txt')
    print(vp_ave, ad_ave, out_ave)

    print('30 - 60 frames: 2nd second')
    vp_ave, ad_ave, out_ave = get_average('60lstm-128-1-tile-acc.txt')
    print(vp_ave, ad_ave, out_ave, vp_ave+ad_ave)
    vp_ave, ad_ave, out_ave = get_average('60-lr_cal-tile-acc.txt')
    print(vp_ave, ad_ave, out_ave)
    vp_ave, ad_ave, out_ave = get_average('60-average-tile-acc.txt')
    print(vp_ave, ad_ave, out_ave)

    print('60 - 90 frames: 3rd second')
    vp_ave, ad_ave, out_ave = get_average('90lstm-128-1-tile-acc.txt')
    print(vp_ave, ad_ave, out_ave, vp_ave+ad_ave)
    vp_ave, ad_ave, out_ave = get_average('90-lr_cal-tile-acc.txt')
    print(vp_ave, ad_ave, out_ave)
    vp_ave, ad_ave, out_ave = get_average('90-average-tile-acc.txt')
    print(vp_ave, ad_ave, out_ave)


if __name__ == "__main__":
    get_average_tile_acc()