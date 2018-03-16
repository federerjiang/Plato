def average(test_sample):
    x = 0.0
    y = 0.0
    z = 0.0
    w = 0.0

    for unit in test_sample:
        x += unit[0]
        y += unit[1]
        z += unit[2]
        w += unit[3]

    sample_length = len(test_sample)
    output = [x/sample_length, y/sample_length, z/sample_length, w/sample_length]

    return output
