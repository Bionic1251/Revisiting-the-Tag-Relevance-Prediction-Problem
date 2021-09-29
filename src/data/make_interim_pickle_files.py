#!/usr/bin/env python
""" Build pickle files raw/ -> interim/

load_rating_sim.py
"""
from pathlib import Path
import os
from loguru import logger
from src.data.imdb import make_imdb_pickle
from src.data.paths import *
from src.data.ratings import make_ratings_pickle_files
from src.data.ratings_sim import make_ratings_sim_pickle
from src.input_data import *
from src.data.make_movies_data_compatible import *
from src.data.make_books_interim import *
from src.data.check_items_ids import *
logger.add('log_make_interim_pickle_files.log')


def run():
    """

    Settings for small test data sets
        :delete_ratings_with_one_vote = False
        :min_doc_rank = small value, e.g. 10

    Default parameters
        :min_doc_rank = 100
        :delete_ratings_with_one_vote = True
        :USE_CACHED_DF_TAGS = False (in settings.py)


    """
    #run_ids_check()
    make_ratings_extract_books()
    make_books_reviews_json()
    make_tags_books()
    make_books_ids()
    make_books_ids2()
    make_books_tag_events()

    make_ratings_pickle_files(FILE_RATINGS_EXTRACT_BOOKS,
                              delete_ratings_with_small_number_of_votes=False)

    make_ratings_sim_pickle(fpath_ratings=EXTRACT_FILE_SURVEY_BOOKS,
                            fpath_ids=EXTRACT_FILE_BOOKS,
                            fpath_ids2=FILE_BOOKS)

    make_imdb_pickle(fpath_tags_survey=EXTRACT_FILE_SURVEY_BOOKS,
                     fpath_survey_movies_ids=EXTRACT_FILE_BOOKS,
                     min_doc_rank=10,
                     fpath_all_movies_ids=FILE_BOOKS,
                     out_pickle_imdb=os.path.join(DIR_PICKLE_FILES, 'imdb.pickle'),
                     out_pickle_imdb_nostem=os.path.join(DIR_PICKLE_FILES, 'imdb_nostem.pickle'))


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    run()

