import pandas as pd
import numpy as np
import os
from data import FEATURES, TARGET


def calc_best_possible_mae(df_input):
    df = df_input[FEATURES + [TARGET]]
    n = len(df)
    df_aggregated = df.groupby(FEATURES).mean().reset_index()
    df_aggregated = df_aggregated.rename(columns={"targets": "targets_mean"})
    df_merged = pd.merge(df, df_aggregated, on=FEATURES, how='left')
    best_possible_mae = (df_merged.targets - df_merged.targets_mean).to_numpy()
    best_possible_mae = np.abs(best_possible_mae)
    best_possible_mae = np.sum(best_possible_mae) / n
    print(best_possible_mae)
    return best_possible_mae


if __name__ == "__main__":
    test0 = pd.read_csv(os.path.join('./temp', "test0.csv"))
    all_data = pd.read_csv('./data/data_survey_with_target.csv')
    calc_best_possible_mae(test0)
    calc_best_possible_mae(all_data)
