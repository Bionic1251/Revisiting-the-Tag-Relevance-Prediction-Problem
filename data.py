import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import gensim
import gensim.downloader
from loguru import logger
from utils_json import *
from sklearn.preprocessing import OneHotEncoder


def split_indices_train_validation(indices, validation_split=0.1):
    n = len(indices)
    validation_len = int(np.round(validation_split * n))
    validation_idx = np.random.choice(indices, size=validation_len, replace=False)
    train_idx = np.array(list(set(indices) - set(validation_idx)))
    return train_idx, validation_idx


def split_data_survey_ten_folds(data_survey_path, save_dir='./temp'):
    """ Prepare 10 train / test splits common for Pytorch and R models """
    data_survey_df = pd.read_csv(data_survey_path)
    # R data: ratings = pd.read_csv(FPATH_Y_R_REPO, header=None)
    # R data: ratings.columns = ['targets']
    # R data: data_survey_df['targets'] = ratings['targets']
    save_path_all = os.path.join(save_dir, 'train_all.csv')
    data_survey_df.to_csv(save_path_all, index=False, sep=",", quotechar='"')
    logger.info(f"Save {save_path_all}")
    for i in range(10):
        save_path_train = os.path.join(save_dir, "train" + str(i) + ".csv")
        save_path_test = os.path.join(save_dir, "test" + str(i) + ".csv")
        dtrain, dtest = train_test_split(data_survey_df, test_size=0.3)
        dtrain.to_csv(save_path_train, index=False, sep=",", quotechar='"')
        dtest.to_csv(save_path_test, index=False, sep=",", quotechar='"')
        logger.info(f"Save {save_path_train}")
        logger.info(f"Save {save_path_test}")


def find_nan_index(df, column_name):
    """
    Find indexes of rows with missing values in data frame column
    """
    nan_index = df[df[column_name].isna()].index
    try:
        nan_index = nan_index[0]
        logger.info(f"Missing value index in the input = {nan_index}")
    except IndexError:
        logger.info(f"No NaN s in column {column_name}")


FPATH_DATA_SURVEY_R_REPO = "movielens-tagnav/r/out/dataSurvey.csv"
FPATH_DATA_SURVEY_LOCAL = "./data/data_survey_with_target.csv"
FPATH_Y_R_REPO = "movielens-tagnav/r/out/y.csv"
R_PREDICTIONS_FILE = "movielens-tagnav/r/out/surveyPredictions.csv"
PYTORCH_PREDICTIONS_FILE = './temp/torch_predictions.txt'
RF_PREDICTIONS_FILE = './temp/rf_predictions.txt'

# We remove / do not select "movieId"
TAG = "tag"
LOG_IMDB = "log_IMDB"
LOG_IMDB_NOSTEM = "log_IMDB_nostem"
RATING_SIMILARITY = "rating_similarity"
AVG_RATING = "avg_rating"
TAG_EXISTS = "tag_exists"
LSI_TAGS_75 = "lsi_tags_75"
LSI_IMDB_175 = "lsi_imdb_175"
TAG_PROB = "tag_prob"
FEATURES = [TAG,
            LOG_IMDB,
            LOG_IMDB_NOSTEM,
            RATING_SIMILARITY,
            AVG_RATING,
            TAG_EXISTS,
            LSI_TAGS_75,
            LSI_IMDB_175,
            TAG_PROB]

TARGET = 'targets'

RE_READ_DATA_FROM_R_REPO = False
if RE_READ_DATA_FROM_R_REPO:
    # READ ALL DATA FROM R repo
    # SET FEATURES AND TARGET
    # SAVE TO ./data/
    data_survey = pd.read_csv(FPATH_DATA_SURVEY_R_REPO)
    data_survey = data_survey[FEATURES]
    tag_value_counts = dict(data_survey['tag'].value_counts())
    # SET TARGET
    targets = pd.read_csv(FPATH_Y_R_REPO, header=None)
    targets.columns = [TARGET]
    data_survey[TARGET] = targets[TARGET]
    data_survey.to_csv(FPATH_DATA_SURVEY_LOCAL, index=False, sep=",", quotechar='"')
else:
    data_survey = pd.read_csv(FPATH_DATA_SURVEY_LOCAL)


FEATURES_NUMERICAL_AND_TARGET = [LOG_IMDB,
                                 LOG_IMDB_NOSTEM,
                                 RATING_SIMILARITY,
                                 AVG_RATING,
                                 TAG_EXISTS,
                                 LSI_TAGS_75,
                                 LSI_IMDB_175,
                                 TAG_PROB,
                                 TARGET]

REDO_TEN_FOLD_DATASETS = True
REDO_TEN_FOLD_DATASETS = False

if REDO_TEN_FOLD_DATASETS:
    split_data_survey_ten_folds(FPATH_DATA_SURVEY_LOCAL, save_dir='./temp')


def aggregate_df_mean_targets(df):
    df_preprocessed = df.copy()
    # df_preprocessed['tag_text'] = df_preprocessed['tag']
    # df_preprocessed['tag'] = df_preprocessed.apply(lambda x: tag_value_counts[x['tag']] / 573.0, axis=1)
    # group_by_columns_ = [c for c in list(df_preprocessed.columns) if c not in ['targets']]
    df_preprocessed = df_preprocessed.groupby(FEATURES).mean().reset_index()
    return df_preprocessed


# train0 = aggregate_df_mean_targets(train0)
# train1 = aggregate_df_mean_targets(train1)
# test0 = test0
# test1 = test1

# REPLACE TAG BY IT'S VALUE COUNTS
# data_survey['tag_text'] = data_survey['tag']
# data_survey['tag'] = data_survey.apply(lambda x: tag_value_counts[x['tag']]/573.0, axis=1)
# data_survey_aggregated['tag_text'] = data_survey_aggregated['tag']
# data_survey_aggregated['tag'] = data_survey_aggregated.apply(lambda x: tag_value_counts[x['tag']]/573.0, axis=1)
#
# === END: REPLACE BY CALL preprocess_df()

# MAKE BINARY TARGET
# data_survey['targets_binary'] = data_survey.apply(lambda x: 0 if x['targets'] < 2 else 1, axis=1)

# SPLIT TRAIN TEST 1 FOLD
train_dat, test_dat = train_test_split(data_survey, test_size=0.3)

LOAD_SCIKIT_DATA = False
if LOAD_SCIKIT_DATA:
    X_train = train_dat[FEATURES]
    Y_train = train_dat[TARGET]
    X_test = test_dat[FEATURES]
    Y_test = test_dat[TARGET]
    X_all = data_survey[FEATURES]
    Y_all = data_survey[TARGET]
    Y_test_targets = np.array(test_dat[TARGET].values.tolist())
    Y_all_targets = np.array(data_survey['targets'].values.tolist())


predictions_r = pd.read_csv(R_PREDICTIONS_FILE, header=None)

# Create and save tags encodings
TAGS_MAPPINGS_FILE_GENSIM = 'temp/tags_word2vec.json'
TAGS_MAPPINGS_FILE_ONE_HOT = 'temp/tags_one_hot.json'

tags = set(data_survey['tag'].values)


def create_and_save_word2vec_tags_mapping():
    vectors = gensim.downloader.load('glove-wiki-gigaword-300')
    save_path = TAGS_MAPPINGS_FILE_GENSIM
    mapping = {}
    for tag in tags:
        for t in tag.split():
            embeddings_for_tag = []
            try:
                embeddings_for_tag.append(vectors[t])
            except KeyError:
                embeddings_for_tag.append(np.ones((300,)) * 0.5)
        mapping[tag] = np.mean(embeddings_for_tag, axis=0).tolist()
    save_json(mapping, save_path)


def load_tags_word2vec_mappings(recalculate_mappings=False):
    if recalculate_mappings:
        create_and_save_word2vec_tags_mapping()
    return load_json(TAGS_MAPPINGS_FILE_GENSIM)


def create_and_save_tags_one_hot_mappings():
    enc = OneHotEncoder()
    mapping = {}
    save_path = TAGS_MAPPINGS_FILE_ONE_HOT
    enc.fit([[e] for e in tags])
    for tag in tags:
        vec = enc.transform([[tag]]).toarray()[0]
        mapping[tag] = list(vec)
    with open(save_path, 'w') as f:
        json.dump(mapping, f)
    logger.info(f"Saved {save_path}")


def load_tags_one_hot_mappings(recalculate_mappings=False):
    if recalculate_mappings:
        create_and_save_tags_one_hot_mappings()
    with open(TAGS_MAPPINGS_FILE_ONE_HOT, 'r') as f:
        data = json.load(f)
        return data


tags_word2vec = load_tags_word2vec_mappings()
tags_one_hot = load_tags_one_hot_mappings()

