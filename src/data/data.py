import pandas as pd
from src.data.paths import *
import csv
import json
from loguru import logger


def books_uid_bid_rating():
    # user_id, book_id, rating
    with open(FILE_RATINGS_EXTRACT_BOOKS, 'r') as f:
        cswr = csv.reader(f, delimiter='\t')
        for row in cswr:
            uid = row[0]
            bid = row[1]
            rating = row[2]
            yield uid, bid, rating

#? try:
#?     #df_movie_data = pd.read_csv(PATH_CSV_MOVIE_DATA, sep='\t')
#?     # Index(['movieId', 'title', 'directedBy', 'starring', 'filmReleaseDate',
#?     #        'vhsReleaseDate', 'dvdReleaseDate', 'dateAdded', 'mpaaRating',
#?     #        'avgRating', 'popularity', 'popRank', 'entropy', 'tstamp', 'lang',
#?     #        'imdbId', 'isClassique', 'movieStatus', 'suggestionId', 'versionNumber',
#?     #        'editDate', 'userId', 'comment', 'rowType', 'noAutoReplace', 'mpaaId',
#?     #        'imdbQuality', 'blankQuality', 'popPercentile', 'popularityLastYear',
#?     #        'popRankLastYear', 'avgRatingLastYear', 'ratingDist'],
#?     #       dtype='object')
#?     #
#?     df_movie_data = pd.read_json(PATH_JSON_MOVIE_DATA)
#?     df_movie_data = df_movie_data.rename(columns={'movieId':'movie_id',
#?         'avgRating':'avg_rating', 'directedBy':'directed_by', 'imdbId':'imdb_id'})
#? except FileNotFoundError as e:
#?     logger.info(f"File not found")
#?     print(e)

with open(PATH_JSON_MOVIE_DATA, 'r') as f:
    ids = []
    starring = []
    directed_by = []
    avg_rating = []
    title = []
    popularity = []
    filmReleaseDate = []
    imdb_id = []
    for row in f:
        cont = json.loads(row)
        ids.append(cont['item_id'])
        title.append(cont['title'])

        try:
            starring.append(cont['starring'])
        except KeyError:
            starring.append('')

        try:
            directed_by.append(cont['directedBy'])
        except KeyError:
            directed_by.append('')

        try:
            avg_rating.append(cont['avgRating'])
        except KeyError:
            avg_rating.append('')


        try:
            imdb_id.append(cont['imdbId'])
        except KeyError:
            imdb_id.append('')
        #popularity.append(cont['popularity'])
        #filmReleaseDate.append(cont['filmReleaseDate'])


df_movie_data = pd.DataFrame({'item_id': ids,
                              'starring': starring,
                              'directed_by':directed_by,
                              'avg_rating': avg_rating,
                              'title': title,
                              'imdb_id': imdb_id,
                              })


def get_books_ids():
    input_file = EXTRACT_FILE_BOOKS
    with open(input_file, 'r') as f:
        cont = f.readlines()
    return list(map(lambda x: int(x), cont))


def movie_ids_movie_data(min_ratings=100):
    """
    Replacing sql query:
    sql = select movieId from %s where popularity >= %d % (TABLE_MOVIE_DATA, minRatings)
    Columns in csv:
    movie_id, title, directed_by, starring, date_added, avg_rating, imdb_id
    TODO: popularity column is missing
    """
    df = df_movie_data[df_movie_data['avg_rating'] > min_ratings]
    movie_ids = df['movie_id'].values.tolist()
    movie_ids = list(set(movie_ids))
    return movie_ids


def sim_sql_query_tag_events():
    df = pd.read_csv(PATH_CSV_TAG_EVENT, sep='\t')
    vals = df.values.tolist()
    return vals


def queries_examples_used_in_project():
    q1 = df_movie_data[['movie_id', 'starring', 'directed_by']].to_numpy().tolist()


def read_reviews_json():
    with open(PATH_JSON_INTERIM_REVIEWS_AGG, 'r') as f:
        return json.load(f)


def get_text_reviews_from_json_obj(reviews_json_obj, item_id):
    try:
        return reviews_json_obj[item_id]['text']
    except KeyError as e:
        pass
    try:
        return reviews_json_obj[str(item_id)]['text']
    except KeyError as e:
        return None


def imdb_corpus():
    with open(PATH_CSV_IMDBCORPUS, newline='') as f:
        csv_reader = csv.reader(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            # movie_id, heading, body
            yield row[0], row[1], row[2]


def imdb_corpus_by_movie_id(query_movie_id):
    for i, row in enumerate(imdb_corpus()):
        movie_id, heading, body = row
        if int(query_movie_id) == int(movie_id):
            return str(movie_id), str(heading), str(body)


if __name__ == "__main__":
    print(df_movie_data)
    #ret = imdb_corpus_by_movie_id(130)
    #print(df_movie_data[df_movie_data['movieId']==130])
    #imdb_req_from_csv = imdb_corpus_by_movie_id(196931)
    #joined_reviews = [" ".join(row) for row in imdb_req_from_csv]
