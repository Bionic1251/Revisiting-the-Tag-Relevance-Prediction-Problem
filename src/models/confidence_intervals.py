import numpy as np
import pandas as pd
import os
from src.data.paths import *
import scipy.stats as st


file_pred_b = os.path.join(PROJECT_DIR, 'temp/predictions_only_books.txt')
file_pred_bm = os.path.join(PROJECT_DIR, 'temp/predictions_books_and_movies.txt')
predictions_b = pd.read_csv(file_pred_b, header=None).to_numpy()
predictions_bm = pd.read_csv(file_pred_bm, header=None).to_numpy()

file_r_pred_b = os.path.join(PROJECT_DIR, 'temp/r_predictions_b.dat')
file_r_pred_bm = os.path.join(PROJECT_DIR, 'temp/r_predictions_bm.dat')
r_predictions_b = pd.read_csv(file_pred_b, header=None).to_numpy()
r_predictions_bm = pd.read_csv(file_pred_bm, header=None).to_numpy()

file_train_books = os.path.join(PROJECT_DIR, 'data/processed/train_test/train_books.csv')
df_train_books = pd.read_csv(file_train_books)
mean_train = df_train_books['targets'].mean()


file_true_values = os.path.join(PROJECT_DIR, 'temp/true_values.txt')
y_true = pd.read_csv(file_true_values, header=None).to_numpy()


def confidence_interval(true_values, predicted):
    n = len(true_values)
    errors = np.abs(y_true - predicted)
    mae = np.mean(errors)
    ci = st.t.interval(0.95, n-1, loc=np.mean(errors), scale=st.sem(errors))
    if 1 == 0:
        print(f"MAE={mae} n = {n} sigma={np.std(errors)}")
    ci_eps = (ci-mae)[1][0]
    return ci_eps

ci_b = confidence_interval(y_true, predictions_b)
ci_bm = confidence_interval(y_true, predictions_bm)
ci_r_b = confidence_interval(y_true, r_predictions_b)
ci_r_bm = confidence_interval(y_true, r_predictions_bm)
ci_avg = confidence_interval(y_true, np.ones(y_true.shape) * mean_train)
print(f"CI = {ci_b} CI DL = {ci_bm}")
print(f"CI R = {ci_r_b} CI R DL = {ci_r_bm}")
print(f"CI AVG = {ci_avg}")
# > CI = 0.012025412058683571 CI DL = 0.012140611370391352
# > CI R = 0.012025412058683571 CI R DL = 0.012140611370391352
# > CI AVG = 0.007820683828471342
