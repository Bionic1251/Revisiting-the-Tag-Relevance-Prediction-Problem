#!/usr/bin/env python
import json
import os
from src.data.paths import *
import csv
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


class TrainTestBooksMovies:
    input_fpath_books = os.path.join(DIR_DATA_PROCESSED, "data_survey_books.csv")
    input_fpath_movies = os.path.join(DIR_DATA_PROCESSED, "data_survey_movies.csv")

    output_books_and_movies_train = os.path.join(DIR_TRAIN_TEST, 'train_books_and_movies.csv')
    output_books_test = os.path.join(DIR_TRAIN_TEST, 'test_books.csv')
    output_books_train = os.path.join(DIR_TRAIN_TEST, 'train_books.csv')
    output_train_all_movies = os.path.join(DIR_TRAIN_TEST, 'train_all_movies.csv')

    output_dir_ten_folds = os.path.join(DIR_TRAIN_TEST, '10folds')

    def __init__(self, books_frac=None):
        self.books_frac=books_frac


    def make_train_test(self, make_ten_folds=False):
        FEATURES_AND_TARGET = ["tag",
                               "log_IMDB",
                               "log_IMDB_nostem",
                               "rating_similarity",
                               "avg_rating",
                               "tag_exists",
                               "lsi_tags_75",
                               "lsi_imdb_175",
                               "tag_prob",
                               "targets"]

        df_movies = pd.read_csv(self.input_fpath_movies, sep=",")
        df_books = pd.read_csv(self.input_fpath_books, sep=",")
        df_movies = df_movies[FEATURES_AND_TARGET]
        df_books = df_books[FEATURES_AND_TARGET]

        if self.books_frac:
            logger.info(f"Take sample from Books dataset")
            df_books = df_books.sample(frac=self.books_frac)
            logger.info(f".. sample df_books.shape = {df_books.shape}")

        train_books, test_books = train_test_split(df_books, test_size=0.3)
        train_books_and_movies = pd.concat([df_movies, train_books])

        train_books_and_movies.to_csv(self.output_books_and_movies_train, index=False)
        train_books.to_csv(self.output_books_train, index=False)
        test_books.to_csv(self.output_books_test, index=False)
        df_movies.to_csv(self.output_train_all_movies, index=False)

        if make_ten_folds:
            kf = KFold(n_splits=10)
            logger.info("Make ten folds for Books")
            fold = 0
            for train_index, test_index in kf.split(df_books):
                fold += 1
                train_books = df_books.iloc[train_index]
                test_books = df_books.iloc[test_index]
                save_path_train = os.path.join(self.output_dir_ten_folds, f'train_books_fold{fold}.csv')
                save_path_test = os.path.join(self.output_dir_ten_folds, f'test_books_fold{fold}.csv')
                train_books.to_csv(save_path_train, index=False)
                test_books.to_csv(save_path_test, index=False)
                logger.info(f"Save into {save_path_train}")
                logger.info(f"Save into {save_path_test}")

    def get_train_test_books_fold(self, fold: int):
        f_train = os.path.join(self.output_dir_ten_folds, f'train_books_fold{fold}.csv')
        f_test = os.path.join(self.output_dir_ten_folds, f'test_books_fold{fold}.csv')
        return pd.read_csv(f_train), pd.read_csv(f_test)

    @property
    def train_books_and_movies(self):
        return pd.read_csv(self.output_books_and_movies_train)

    @property
    def train_books(self):
        return pd.read_csv(self.output_books_train)

    @property
    def test_books(self):
        return pd.read_csv(self.output_books_test)

    @property
    def train_movies(self):
        return pd.read_csv(self.output_train_all_movies)


if __name__ == "__main__":
    TrainTestBooksMovies().make_train_test()
