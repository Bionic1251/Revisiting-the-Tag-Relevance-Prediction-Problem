#!/usr/bin/env python
"""
Create movies data in *.jsonl format like books.
"""
import gzip
import shutil
import numpy as np
import pdb
import pandas as pd
import os
import json
from loguru import logger
from src.data.paths import *


OUTPUT_FOLDER = "/home/ms314/projects/tagnav-code-refactored/data/raw/movies_formatted"


def make_ratings():
    input_file = os.path.join(DIR_DATA_RAW, 'movies_old/ratings_extract.txt')
    output_file = os.path.join(OUTPUT_FOLDER, 'ratings.json')
    with open(input_file, 'r') as f, open(output_file, 'w') as fout:
        for _, row in enumerate(f):
            c1, c2, c3 = row.split()
            uid, movie_id, rating = int(c1), int(c2), float(c3)
            fout.write(json.dumps({'item_id': movie_id, 'user_id': uid, 'rating': rating}) + '\n')
    logger.info(f"Save into {output_file}")


def make_tags():
    """
    Output format
    {"tag": "18th century", "id": 0}
    {"tag": "1920s", "id": 1}
    """
    f_tags = os.path.join(DIR_DATA_RAW, 'movies_old/survey/tags')
    f_tags_survey = os.path.join(DIR_DATA_RAW, 'movies_old/survey/tags_survey')
    f_mainanswers = os.path.join(DIR_DATA_RAW, 'movies_old/survey/mainanswers.txt')
    f_tag_event = os.path.join(DIR_DATA_RAW, 'movies_old/json_data/out_csv/tag_event.csv')
    output_file = os.path.join(OUTPUT_FOLDER, 'tags.json')
    tags = [e for e in open(f_tags, 'r').read().splitlines()]
    tags_survey = [e for e in open(f_tags_survey, 'r').read().splitlines()]
    tags_tagevent = pd.read_csv(f_tag_event, sep='\t')['tag'].to_numpy().tolist()
    tags_tagevent = set(tags_tagevent)
    tags_mainanswers = []
    with open(f_mainanswers, 'r') as f:
        for row in f:
            row_s = row.split('\t')
            tags_mainanswers.append(row_s[3])
    tags = set(tags)
    tags_survey = set(tags_survey)
    tags_mainanswers = set(tags_mainanswers)
    all_tags = tags_mainanswers | tags_survey | tags | tags_tagevent
    tags = {}
    with open(output_file, 'w') as f:
        for k, tag in enumerate(all_tags):
            f.write(json.dumps({'tag': tag, 'id': k}) + '\n')
    logger.info(f"Save into {output_file}")


def make_tag_count():
    """
    format:
    {"book_id": 115, "tag_id": 13, "num": 52}
    {"book_id": 115, "tag_id": 25, "num": 180}
    """
    # userId, movieId, rating
    input_file = os.path.join(DIR_DATA_RAW, 'movies_old/json_data/out_csv/tag_event.csv')
    output_file = os.path.join(OUTPUT_FOLDER, 'tag_count.json')
    df = pd.read_csv(input_file, sep='\t')
    df = df.to_numpy()
    tags_ids = get_tag_ids()
    with open(output_file, 'w') as f:
        for row in df:
            item_id = row[0]
            tag = row[1]
            count = row[2]
            tag_id = tags_ids[tag]
            f.write(json.dumps({'item_id': item_id, 'tag_id': tag_id, 'num': count}) + '\n')
    logger.info(f"Save into {output_file}")


def get_tag_ids():
    """
    {"tag": "18th century", "id": 0}
    """
    input_file = os.path.join(OUTPUT_FOLDER, 'tags.json')
    mapping = {}
    with open(input_file, 'r') as f:
        for row in f:
            cont = json.loads(row)
            mapping[cont['tag']] = cont['id']
    return mapping


def make_metadata():
    """
    {"id": 16827462,
     "url": "https://www.goodreads.com",
     "title": "The Fault in Our Stars by John Green",
     "book_name": "The Fault in Our Stars", "authors": "John Green",
     "lang": "eng",
     "img": "https://images.gr-assets.com/books/1360206420m/11870085.jpg",
     "year": 2012,
     "description": "There is .....",
     "pop": 20663}

    """
    # {"movieId": 1,
    #  "title": "Toy Story (1995)",
    #  "directedBy": "John Lasseter",
    #  "starring": "Tim Allen, Tom Hanks, Don Rickles, Jim Varney, John Ratzenberger, Wallace Shawn, Laurie Metcalf, John Morris, R. Lee Ermey, Annie Potts",
    # "dateAdded": null,
    # "avgRating": 3.89146,
    # "imdbId": "0114709"}
    input_file = os.path.join(PROJECT_DIR, "data/raw/movies_old/json_data/movie_data.json")
    output_file = os.path.join(OUTPUT_FOLDER, 'metadata.json')
    tot = 0
    with open(input_file, 'r') as f, open(output_file, 'w') as fout:
        cont = json.load(f)
        for item in cont:
            item['item_id'] = item['movieId']
            del item['movieId']
            fout.write(json.dumps(item)+'\n')
    logger.info(f"Save into {output_file}")


def make_mainanswers():
    """
    uid,book_id,tag_id,score
    2,604666,151,5
    2,1222101,151,5
    2,1272463,151,5
    """
    input_file = os.path.join(PROJECT_DIR, "data/raw/movies_old/survey/mainanswers.txt")
    output_file = os.path.join(OUTPUT_FOLDER, 'mainanswers.csv')
    # uid, roundIndex, movieId, tag, response
    df_input = pd.read_csv(input_file, sep='\t', header=None)
    df_input.columns = ['uid', 'roundIndex', 'item_id', 'tag', 'score']
    print(df_input)
    mapping = get_tag_ids()
    df = pd.DataFrame({'uid': df_input['uid'],
                       'item_id': df_input['item_id'],
                       'tag_id': [mapping[e] for e in df_input['tag']],
                       'score': df_input['score']})
    df.to_csv(output_file, sep=',', index=False)
    logger.info(f"Save into {output_file}")


def make_reviews():
    """
    Output format
    {"book_id": 41335427,
     "user_id": 0,
     "txt": "Annual re-read 2013. I liked HBP much more this time. Annual re-read 2017."}
    """
    # {"movieId": 172063,
    #  "heading": "one-shot record of a belly dancer",
    #  "body": "Carmen"}
    input_file = '/home/ms314/datasets/tagnav/json_data/imdbcorpus_out.json'
    output_file = os.path.join(OUTPUT_FOLDER, 'reviews.json')
    with open(input_file, 'r') as f, open(output_file, 'w') as fout:
        for line in f:
            cont = json.loads(line)
            fout.write(json.dumps({'item_id': cont['movieId'], 'txt': cont['heading'] + '; ' + cont['body']}) + '\n')
    logger.info(f"Save into {output_file}")



if __name__ == "__main__":
    make_ratings()
    make_tags()
    make_tag_count()
    make_mainanswers()
    make_reviews()
    make_metadata()

