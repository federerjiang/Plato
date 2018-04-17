import os

train_folder = 'train'
test_folder = 'test'

for root, dirname, filenames in os.walk(train_folder):
    count = 0
    for filename in filenames:
        pass
        # count += 1
        # print(filename)
        # print(dirname)
    # print(count)