#!/usr/bin/env python
import pandas as pd
from sklearn.model_selection import KFold
import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
from loguru import logger

def make_10folds():
    logger.info("Make 10 folds")
    input_file = os.path.join(os.getenv('PROJECT_DIR'), 'data/processed/features_r.csv')
    output_folder = os.path.join(os.getenv('PROJECT_DIR'), 'data/processed/10folds')
    #df = pd.read_csv("../../../data_survey_with_target.txt")
    df = pd.read_csv(input_file)
    kf10 = KFold(n_splits=10, shuffle=True, random_state=1)
    i = 0
    for train_index, test_index in kf10.split(df):
        train_file = os.path.join(output_folder, "train" + str(i) + ".csv")
        test_file = os.path.join(output_folder, "test" + str(i) + ".csv")
        df.iloc[train_index].to_csv(train_file, index=False)
        df.iloc[test_index].to_csv(test_file, index=False)
        i += 1
        logger.info(f"Save {train_file}")
        logger.info(f"Save {test_file}")


if __name__ == "__main__":
    make_10folds()
