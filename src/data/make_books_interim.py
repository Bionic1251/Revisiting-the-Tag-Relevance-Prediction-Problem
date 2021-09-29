#!/usr/bin/env python
import json
import os
from numpy import log
from src.data.paths import *
import csv
from loguru import logger
import pandas as pd
from tqdm import tqdm
from src.settings import USE_CACHED_DF_TAGS


def test_books_reviews_json(test_id=4):
    logger.info("Run test")
    input_file = os.path.join(DIR_DATA_RAW, 'reviews.json')
    output_file = PATH_JSON_INTERIM_BOOKSCORPUS
    out = {}
    with open(input_file, 'r') as f:
        for i, row in enumerate(f):
            cont = json.loads(row)
            b_id = cont['item_id']
            if b_id == test_id:
                logger.info(b_id)


def make_books_reviews_json():
    """ Map id: all reviews """
    input_file = os.path.join(DIR_DATA_RAW, 'reviews.json')
    output_file = PATH_JSON_INTERIM_REVIEWS_AGG
    logger.info(f"Input file {input_file}")
    out = {}
    with open(input_file, 'r') as f:
        for _, row in enumerate(f):
            cont = json.loads(row)
            b_id = cont['item_id']
            txt = cont['txt']
            txt = txt.replace('\t', ' ')
            if b_id not in out:
                out[b_id] = {'text': txt}
            else:
                out[b_id]['text'] = out[b_id]['text'] + ' ' + txt
    with open(output_file, 'w') as f:
        json.dump(out, f, indent=4)
    logger.info(f"Output file {output_file}")


def make_books_ids2():
    """
    FILE_MOVIES = os.path.join(DIR_DATA_RAW, "movies.txt")
    ->
    FILE_BOOKS = os.path.join(DIR_DATA_RAW, "books.txt")
    """
    input_file = os.path.join(DIR_DATA_RAW, 'ratings.json')
    output_file = FILE_BOOKS
    books_ids = set()
    with open(input_file, 'r') as f:
        for row in f:
            cont = json.loads(row)
            book_id = cont['item_id']
            books_ids.add(book_id)
    with open(output_file, 'w') as fout:
        for b_id in books_ids:
            fout.write(str(b_id) + '\n')

    logger.info(f"Save into {output_file}")


def make_books_ids():
    """
    EXTRACT_FILE = os.path.join(DIR_DATA_RAW, 'survey/movies')
    ->
    EXTRACT_FILE_BOOKS = os.path.join(DIR_DATA_RAW, 'books')
    """
    # todo: check if book ids are also in other files
    input_file = os.path.join(DIR_DATA_RAW, 'ratings.json')
    output_file = EXTRACT_FILE_BOOKS
    books_ids = set()
    with open(input_file, 'r') as f:
        for row in f:
            cont = json.loads(row)
            book_id = cont['item_id']
            books_ids.add(book_id)
    with open(output_file, 'w') as fout:
        for b_id in books_ids:
            fout.write(str(b_id) + '\n')
    logger.info(f"Save into {output_file}")


def make_df_tags(use_cached=False):
    input_file = FILE_TAGS_BOOKS_JSON
    output_file = os.path.join(DIR_DATA_INTERIM, 'tags.csv')
    if use_cached:
        logger.info("WARNING: Cached file is used for tags. Turn off when running on new data.")
    if not use_cached:
        logger.info("WARNING: Recalculate cache file for tags. Turn on to use cached.")
        logger.info(f"Make cache file from: {input_file}")
        df = pd.DataFrame(columns=['tag_id', 'tag'])
        with open(input_file, 'r') as f:
            for i, row in enumerate(f):
                cont = json.loads(row)
                tag_text = cont['tag']
                tid = cont['id']
                df.loc[i] = [tid, tag_text]
        df.to_csv(output_file, index=False)
        logger.info(f"Save into {output_file}")
    else:
        logger.info(f"Read from cache: {input_file}")
        df = pd.read_csv(output_file)
    return df


df_tags = make_df_tags(use_cached=USE_CACHED_DF_TAGS)


def make_tags_books():
    """ survey/tags_survey """
    # input_file = os.path.join(DIR_DATA_RAW, "tag_count.json")
    logger.info(f"Start make_tags_books()")
    output_file = EXTRACT_FILE_SURVEY_BOOKS
    df_answers = read_mainanswers()
    r = pd.merge(df_answers, df_tags, how='left', left_on='tag_id', right_on='tag_id')
    tags = r['tag'].unique()
    with open(output_file, 'w') as fout:
        for tag in tags:
            fout.write(str(tag) + '\n')
    logger.info(f"Save into {output_file}")


def get_tag_by_id(tag_id):
    input_file = aupath(DIR_DATA_RAW, 'tags.json')
    with open(input_file, 'r') as f:
        for row in f:
            cont = json.loads(row)
            tag_text = cont['tag']
            tid = cont['id']
            if str(tag_id) == str(tid):
                return tag_text


def make_books_tag_events():
    """ select movieId, lower(tag), count(*) from tag_events .. """
    input_file = os.path.join(DIR_DATA_RAW, "tag_count.json")
    output_file = FILE_TAG_EVENTS_BOOKS
    # global: df_tags = make_df_tags()
    with open(input_file, 'r') as f, open(output_file, 'w') as fout:
        csv_writer = csv.writer(fout, delimiter='\t')
        for i, row in enumerate(f):
            cont = json.loads(row)
            b_id = cont['item_id']
            tag_id = cont['tag_id']
            num = cont['num']
            tag_text = df_tags[df_tags['tag_id'] == tag_id]['tag'].values
            tag_text = ''.join(tag_text)
            csv_writer.writerow([b_id, tag_text, num])
    logger.info(f"Save into {output_file}")


def make_ratings_extract_books():
    """
    ratings_extract.txt
    Compatible data format
    userId, movieId, rating
    """
    input_file = os.path.join(DIR_DATA_RAW, "ratings.json")
    output_file = FILE_RATINGS_EXTRACT_BOOKS
    # os.path.join(DIR_DATA_RAW, 'ratings_extract_books.txt')
    with open(input_file, 'r') as f, open(output_file, 'w') as outf:
        csvw = csv.writer(outf, delimiter='\t')
        for row in f:
            cont = json.loads(row)
            user_id = cont['user_id']
            book_id = cont['item_id']
            rating = cont['rating']
            csvw.writerow([user_id, book_id, rating])
    logger.info(f"Save into {output_file}")


def read_mainanswers():
    mainanswers = []
    with open(FILE_MAINANSWERS_BOOKS, "r") as reader:
        line = reader.readline()
        while line != "":
            obj = json.loads(line)
            mainanswers.append(obj)
            line = reader.readline()
    mainanswers_df = pd.DataFrame(mainanswers)
    mainanswers_df = mainanswers_df.rename(columns={"user_id":"uid"})
    return mainanswers_df


def make_mainanswers_compatible(output_file=FILE_BOOKS_MAINANSWERS_INTERIM):
    """
    Make this format

    survey/mainanswers.txt
    173626	0	3108	mission from god	3
    173626	0	2858	mission from god	1

    userId     = int(vals[0])
    roundIndex = int(vals[1])
    movieId    = int(vals[2])
    tag        = vals[3]
    response   = int(vals[4])
    """
    logger.info("Make mainanswers books compatible")
    df = read_mainanswers()
    df['round_index'] = 0
    tags_books = df_tags  # make_df_tags()
    df = pd.merge(df, tags_books, on='tag_id', how='left')
    df = df[['uid', 'round_index', 'item_id', 'tag', 'score']]
    df.to_csv(output_file, header=False, index=False, sep='\t')
    logger.info(f"Save into {output_file}")


if __name__ == "__main__":
    make_ratings_extract_books()
    # make_tags_books()
    # make_books_ids()
    # make_books_ids2()
    # make_books_tag_events()
    # make_books_reviews_json()
    # test_books_reviews_json()
    # make_mainanswers_compatible()
    # make_books_tag_events()
