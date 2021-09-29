from src.data.make_processed_train_test import TrainTestBooksMovies
import numpy as np
import torch
from torch import nn


def calc_loss_from_arrays(true_values, predictions, r_predictions_file=None):
    predictions_pytorch = torch.tensor(predictions, dtype=torch.float)
    true_values = torch.tensor(true_values, dtype=torch.float)
    fn_loss = nn.L1Loss()
    mae = fn_loss(true_values, predictions_pytorch)
    return mae


def calc_average_mae():
    trtst = TrainTestBooksMovies()
    trtst.make_train_test(make_ten_folds=False)
    df_train_m = trtst.train_movies
    df_train_b = trtst.train_books
    df_train_bm = trtst.train_books_and_movies
    df_test = trtst.test_books
    avg_response_b = df_train_b['targets'].mean()
    predictions = np.zeros(df_test['targets'].shape[0])
    predictions = avg_response_b * predictions
    y_true = df_test['targets'].to_numpy()
    mae_avg = calc_loss_from_arrays(predictions, y_true)
    print(mae_avg)


if __name__ == "__main__":
    calc_average_mae()
    # -> tensor(2.2069)
