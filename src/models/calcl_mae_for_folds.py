#!/usr/bin/env python
import numpy as np
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import mean_absolute_error
load_dotenv(find_dotenv())
from src.common import *
import pandas as pd
from loguru import logger

DIR_PREDICTIONS = get_path('data/predictions')
DIR_PROCESSED = get_path('data/processed/10folds')

def fn_mae(y_pred, y_true):
    test = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    assert mae == test
    return mae

def calc_mae_for_ten_folds():
    save_path = get_path('performance_results_tenfolds.txt', base_dir=DIR_PREDICTIONS)
    with open(save_path, "w") as f:
        for i in range(10):
            r_predictions_path = get_path("r_predictions_fold_"+str(i)+".txt", base_dir=DIR_PREDICTIONS)
            predictions_path = get_path("predictions_fold_" + str(i) + ".txt", base_dir=DIR_PREDICTIONS)
            y_true_path = get_path("test"+str(i)+".csv", base_dir=DIR_PROCESSED)
            y_true = pd.read_csv(y_true_path, sep=',')['targets'].to_numpy()
            r_predictions = pd.read_csv(r_predictions_path, header=None).to_numpy().squeeze()
            predictions = pd.read_csv(predictions_path, header=None).to_numpy().squeeze()
            assert y_true.shape == r_predictions.shape
            assert y_true.shape == predictions.shape
            mae_r = fn_mae(r_predictions, y_true)
            mae = fn_mae(predictions, y_true)
            perf_msg = f"Fold {i}; MAE PyTorch (TagDL): {mae}; R MAE (Glmer): {mae_r};"
            logger.info(perf_msg)
            f.write(perf_msg + '\n')
    logger.info(f"Save performance results for 10 folds into {save_path}")


if __name__ == "__main__":
    calc_mae_for_ten_folds()
