class Args:
    def __init__(self, test_label_length=30):
        self.train_sample_length = 30
        self.label_sample_length = 30  # for TrainDataLoader
        self.test_label_length = test_label_length  # for TestDataLoader
        self.batch_size = 32  # for CudaDataLoader
        self.tile_column = 12
        self.tile_row = 6
        self.vp_length = 110
        self.vp_height = 90
        self.ad_length = 170
        self.ad_height = 110


if __name__ == "__main__":
    args = Args(test_label_length=30)
    # print(args.batch_size)