#!/usr/bin/env python

# train_relevance.txt sample
# tag  movieId  userId  survey_response rank sample_percentile tag_exists tag_count tag_movie_rating_similarity term_freq_IMDB_log lsi_imdb_25 lsi_imdb_175 term_freq_IMDB_log_nostem  avg_movie_rating   lsi_tags_75 
# motorcycle	4011	167627	1	8411	0.7151	0	0	0.06859	 -0.3818  -0.1022    -0.0315	-0.3999  4.069	 0.1174
# motorcycle	2	167627	1	11761	0.9999	0	0	0.06198	 2.13	  -0.002316  0.02902    2.39	 3.266	 -0.1404
#
# predict_relevance_sample.csv
# tag,movieId,tag_exists,tag_count,tag_movie_rating_similarity,term_freq_IMDB_log,lsi_imdb_25,lsi_imdb_175,term_freq_IMDB_log_nostem,avg_movie_rating,lsi_tags_75
# motorcycle,1,0,0,0.1131,-0.3798,0.1266,-0.1511,-0.3964,3.891,-0.08793
# motorcycle,32770,0,0,-0.1247,-0.3321,-0.4298,-0.0451,-0.3246,3.582,-0.09302
# motorcycle,196611,0,0,-0.08125,0.8005,0.0,0.0,0.791,3.051,0.0
# ...

from src.data.paths import *
import random
import pandas as pd
from loguru import logger
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import time
import gc
import pickle as cPickle
from src.data.data import *
from src.common import Input
from src.features.termfreq import *
from src.features.featureoptions import *
import src.features.lsi as lsi
import src.features as features
from src.features.mainsurvey.select_tags import getTagsFromFile
from src.features.build_features import getSurveyTagsFromFile
from src.features.build_features import getSurveyMoviesFromFile
from src.features.build_features import SurveyData
from src.features.build_features import getSurveyTagsFromFile, getSurveyMoviesFromFile
from src.features.build_features import TagEvents, TagCount, TagExists
from src.features.build_features import FeatureResponseExtract, FeatureExtract, RatingSimilarityFeature
from src.data.data import movie_ids_movie_data
from src.data.ratings_sim import make_ratings_sim_obj
from src.data.ratings import make_ratings_obj, make_ratings_obj_adj
from src.data.imdb import getMoviesFromFile, make_imdb_obj, make_imdb_obj_nostem

def create_foldere_if_not_exists(folder_path):
    try:
        Path(folder_path).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder {folder_path} already exists")

create_foldere_if_not_exists(os.path.join(PROJECT_DIR, 'logs'))
logger.add(os.path.join(PROJECT_DIR, 'logs/log_test_movies_features.log'))

def aupath(folder, fname):
    return os.path.join(folder, fname)


DIR_TEST_DATA = os.path.join(PROJECT_DIR, 'src/tests/test_data/')
FILE_TRAIN_REL = os.path.join(DIR_TEST_DATA, 'train_relevance_sample.csv') 
FILE_PRED_REL = os.path.join(DIR_TEST_DATA, 'predict_relevance_sample.csv')

df_main_answers = pd.read_csv(aupath(DIR_TEST_DATA, 'mainanswers.txt'), sep='\t',names=["user_id", "unknown", "movie_id", "tag", "rating"])
df_train_rel = pd.read_csv(FILE_TRAIN_REL, sep='\t')
df_pred_rel = pd.read_csv(FILE_PRED_REL, sep='\t')


def movies_tag_events():
    """ tag_event file sample
    movie_id	tag	count
    1    	1990s	1
    1    	2009	1
    1    	3d	1

    """
    df = pd.read_csv(aupath(DIR_TEST_DATA, 'tag_event.csv'), sep='\t')
    vals = df.values.tolist()
    return vals


class TestMoviesFeatures:
    def test_run(self, 
                 tags_input=['motorcycle', '70mm', 'ninja'], 
                 #tags_input=['motorcycle'], 
                 items_ids_input=[2, 4011]):

        limit_movies = 1

        # Set up IDs
        if tags_input is None:
            survey_tags = getSurveyTagsFromFile(fpath=EXTRACT_FILE_SURVEY)
        else:
            survey_tags = tags_input 
        
        if items_ids_input is None:
            survey_movies = getSurveyMoviesFromFile(fpath=EXTRACT_FILE)
        else:
            survey_movies = items_ids_input

        all_items_ids = getMoviesFromFile(fpath=aupath(DIR_DATA_RAW, 'movies_old/movies.txt'))

        if limit_movies:
            logger.info("Limit movies")
            all_items_ids = random.sample(all_items_ids, limit_movies)
            all_items_ids.extend(items_ids_input)

        tags = getSurveyTagsFromFile(fpath=aupath(DIR_TEST_DATA, 'tags_survey'))
        tags = [e for e in tags if e in tags_input]
        logger.info(f"tags={tags}")
        # rename movies
        main_survey_data = SurveyData(DIR_TEST_DATA + '/mainanswers.txt', 
                                      excludeAdditionalRounds=False, 
                                      excludeNotSure=True, 
                                      excludeAllEqual=True, 
                                      include_tags=tags)

        logger.info("Set up features containers")
        items_ids_survey = getSurveyMoviesFromFile(fpath=aupath(DIR_TEST_DATA, 'movies_survey'))
        items_ids_all = movie_ids_movie_data(min_ratings=3)
        items_ids_all = items_ids_input
        logger.info(f"Number of survey items = {len(items_ids_survey)}")
        logger.info(f"Number of all items = {len(items_ids_all)}")
        featureExtract = FeatureExtract(tags, items_ids=items_ids_survey)
        featureResponseExtract = FeatureResponseExtract(main_survey_data, items_ids=items_ids_all)

        logger.info("Calculate tag exists and counts features")
        tag_events = movies_tag_events()
        logger.info(".. create TagEvents() object")
        tagEvents = TagEvents(includeTags=tags, items_ids_tags_counts_list=tag_events)
        logger.info(".. create TagExists() object")
        tagExistsFeature = TagExists(tagEvents)
        logger.info(".. create TagCount() object")
        tagCountFeature = TagCount(tagEvents)
        logger.info(".. add features to featureExtract() object")
        featureExtract.addFeatures([tagExistsFeature, tagCountFeature])
        logger.info(".. add features to featureResponseExtract() object")
        featureResponseExtract.addFeatures([tagExistsFeature, tagCountFeature])
        del tagExistsFeature, tagCountFeature
        del tagEvents
        gc.collect()

        logger.info("Calculate rating similarity feature")
        #
        # ratingSim = cPickle.load(open(FILE_RATING_SIM, 'rb'))
        # ratings_extract file:
        # user_id, item_id, rating
        # 1	5	3
        # 1	10	4
        # 1	13	4
        # ratingSim, ratingSimAdj = 
        # write_r_extracts.py
        ratings_obj_adj = make_ratings_obj_adj()

        ratingSim = make_ratings_sim_obj(ratings_obj_adj, 
                                         survey_tags=survey_tags, 
                                         survey_movies=survey_movies,
                                         application_movies=all_items_ids)
        ratingSimFeature = RatingSimilarityFeature(ratingSim)
        featureExtract.addFeature(ratingSimFeature)
        featureResponseExtract.addFeature(ratingSimFeature)
        del ratingSimFeature
        del ratingSim
        gc.collect()


        logger.info(f"Calculate IMDB log term frequency feature")
        # imdbCorpus = cPickle.load(open(FILE_IMDB_CORPUS, 'rb'),  encoding='latin1')

        imdbCorpus = make_imdb_obj(survey_tags=survey_tags, 
                                   survey_movies=survey_movies,
                                   application_movies=all_items_ids, 
                                   m_or_b=Input.MOVIES, 
                                   min_doc_rank=10)

        logImdbFeature = TermFreq(imdbCorpus, 
                                  transform=LogTransformRelativeBeta3(imdbCorpus, tags, c=1), 
                                  qualifier="log")
        
        featureExtract.addFeature(logImdbFeature)
        featureResponseExtract.addFeature(logImdbFeature)
        del logImdbFeature
        gc.collect()

        logger.info("Calculate IMDB corpus SVD features")
        cvs = lsi.CorpusVectorSpace(imdbCorpus)
        for svdRank in [25, 175]:
            # Compute SVD of IMDB Corpus
            corpusSVD = lsi.SVD(cvs, svdRank=svdRank, name="_corpus_%d"%svdRank)
            corpusSVD.load()

            # Add feature for this
            lsiFeature = features.LsiFeature(corpusSVD, qualifier="imdb_" + str(svdRank))
            featureExtract.addFeature(lsiFeature)
            featureResponseExtract.addFeature(lsiFeature)
            del lsiFeature
            del corpusSVD
            gc.collect()
        del cvs
        del imdbCorpus
        gc.collect()


        logger.info("Set up IMDB no-stem log term frequency feature")
        #
        #imdbNoStemCorpus = cPickle.load(open(FILE_IMDB_NOSTEM_CORPUS, 'rb'),  encoding='latin1')
        imdbNoStemCorpus = make_imdb_obj_nostem(survey_tags=survey_tags, 
                                                survey_movies=survey_movies,
                                                application_movies=all_items_ids, 
                                                m_or_b=Input.MOVIES, 
                                                min_doc_rank=10)
        imdbNoStemFeature = TermFreq(imdbNoStemCorpus, 
                                     transform=LogTransformRelativeBeta3(imdbNoStemCorpus, tags, c=1),
                                     qualifier="log_nostem")
        featureExtract.addFeature(imdbNoStemFeature)
        featureResponseExtract.addFeature(imdbNoStemFeature)
        del imdbNoStemFeature
        del imdbNoStemCorpus
        gc.collect()

        logger.info("Set up average rating feature")
        #
        #ratingsData = cPickle.load(open(FILE_RATINGS, 'rb'),  encoding='latin1')
        ratingsData = make_ratings_obj(ratings_file=FILE_RATINGS_EXTRACT)
        avgRatingFeature = features.AvgRating(ratingsData)
        featureExtract.addFeature(avgRatingFeature)
        featureResponseExtract.addFeature(avgRatingFeature)
        del avgRatingFeature
        del ratingsData
        gc.collect()

        logger.info("Compute tag event SVD, and set up tag lsi feature")
        #
        #tagEvents = TagEvents(includeTags=mainsurvey.select_tags.getTagsFromFile())
        tagEvents = TagEvents(includeTags=tags, m_or_b=Input.BOOKS)
        tvs = lsi.TagVectorSpace(tagEvents)
        svdRank = 75
        tagSVD = lsi.SVD(tvs, svdRank=svdRank, name="_tags_%d"%svdRank)
        tagSVD.load()
        tagLsiFeature = features.LsiFeature(tagSVD, qualifier='tags_' + str(svdRank))
        featureExtract.addFeature(tagLsiFeature)
        featureResponseExtract.addFeature(tagLsiFeature)
        del tagLsiFeature
        del tagSVD
        del tagEvents
        gc.collect()

        logger.info("Write the extract files for the R process")
        create_foldere_if_not_exists(os.path.join(PROJECT_DIR, 'temp'))
        FILE_DATA_PREDICT_RELEVANCE = os.path.join(PROJECT_DIR, 'temp/predict_rel.txt')
        FILE_DATA_TRAIN_RELEVANCE = os.path.join(PROJECT_DIR, 'temp/train_rel.txt')
        featureExtract.writeExtract(FILE_DATA_PREDICT_RELEVANCE)
        featureResponseExtract.writeExtract(FILE_DATA_TRAIN_RELEVANCE )
        logger.info(f"Save int {FILE_DATA_PREDICT_RELEVANCE}")
        logger.info(f"Save int {FILE_DATA_TRAIN_RELEVANCE}")


class MakeTestData:
    
    file_pred_rel="/home/ms314/datasets/tagnav/ml-tagnav-files/temp/r/data/predict_relevance.txt"
    file_train_rel="/home/ms314/datasets/tagnav/ml-tagnav-files/temp/r/data/train_relevance.txt"

    def make_test_data_samples(self):
        make_test_data_train_rel() 
        make_test_sample_pred_rel()


    def make_test_sample_pred_rel(self):
        save_path = FILE_PRED_REL
        df_pred_rel = pd.read_csv(self.file_pred_rel, sep='\t')
        sample = df_pred_rel[df_pred_rel['tag']=='motorcycle']
        sample.to_csv(save_path, index=False, sep='\t')
        logger.info(f"Save into {save_path}")

    def make_test_data_train_rel(self):
        """ found: motorcycle : 2 """
        save_path = FILE_TRAIN_REL
        df_train_relevance = pd.read_csv(self.file_train_rel, sep='\t')
        print(df_train_relevance['tag'].value_counts().tail(10))
        #sample = df_train_relevance[df_train_relevance['tag']=='motorcycle']
        #sample.to_csv(save_path, index=False, sep='\t')
        #logger.info(f"Save into {save_path}")


if __name__ == "__main__":
    t_obj = TestMoviesFeatures()
    t_obj.test_run()
    # m = MakeTestData()
    # m.make_test_data_train_rel()

