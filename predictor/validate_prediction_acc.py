from validate_prediction_loss import lstm_predict
from validate_prediction_loss import other_predict
from average import average
from lr import lr
from cuda_lstm import LSTMPredict
from data_loader import TestDataLoader


def validate_rotation_acc(test_data_loader):
    for inputs, label in test_data_loader:



def validate_tile_acc(test_data_loader):
    pass


if __name__ == "__main__":
    test_data_loader = TestDataLoader()


