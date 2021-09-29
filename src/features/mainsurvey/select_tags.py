"""
This module handles the survey tags.
"""

import os.path
from loguru import logger
#from ml import tagset
from src.data.paths import * # EXTRACT_FILE_BOOKS, EXTRACT_FILE_SURVEY_BOOKS, DIR_SURVEY
from src.common import Input


# Default values for building the survey
MIN_USERS = 10
MIN_PERCENTILE = .05
EXCLUDE_TAG_IF_ALL_STOP_WORDS = True

# The tag survey files
#EXTRACT_FILE_SURVEY_MOVIES = os.path.join(DIR_SURVEY, 'tags_survey')
#EXTRACT_FILE_MOVIES = os.path.join(DIR_SURVEY, 'tags')

EXTRACT_FILE = EXTRACT_FILE_BOOKS
EXTRACT_FILE_SURVEY = EXTRACT_FILE_SURVEY_BOOKS

class TagSet:

    def __init__(self, minUsers=10, minPercentile=0, excludeActors=True, excludeDirectors=True, excludeTagIfAllStopWords=True):
        mlcnx = sql_cnx.getMLConnection()
        mlcursor = mlcnx.cursor()

        self.tags = set([])
        self.numUsers = {}
        self.numEvents = {}
        if minPercentile > 0:
            self.thumbRatings = thumbratings.ThumbRatings()
        else:
            self.thumbRatings = None
        if excludeTagIfAllStopWords:
            stopWords = stopwords.getStopWords()


        md = metadata.MetaData()

        sql = """
            select lower(tag), count(distinct userId) as numUsers, count(*) from tag_events group by lower(tag) having count(distinct userId) >= %d order by numUsers desc
        """%minUsers
        mlcursor.execute(sql)
        for tag, numUsers, numEvents in mlcursor.fetchall():
            if excludeTagIfAllStopWords:
                wordsInTag = set(re.sub(r"[^A-Za-z0-9]", " ", tag).split())
                if len(wordsInTag - stopWords)==0:
                    continue
            if (excludeActors and md.isActor(tag)) or (excludeDirectors and md.isDirector(tag)):
                continue
            if minPercentile > 0 and self.thumbRatings.getPercentile(tag) <= minPercentile:
                continue
            self.tags.add(tag)
            self.numUsers[tag]=numUsers
            self.numEvents[tag]=numEvents

    def display(self):
        usersTags = [(self.numUsers[tag], tag) for tag in self.tags]
        usersTags.sort()
        usersTags.reverse()
        print(string.join(['%s (%d)'%(tag, numUsers) for numUsers, tag in usersTags],'\n'))

    def reportThumbRatingFilter(self):
        if not self.thumbRatings:
            self.thumbRatings = thumbratings.ThumbRatings()
        saveDiscard = set([])
        for threshold in [.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1]:
            discard = set([tag for tag in self.tags if self.thumbRatings.getPercentile(tag) <= threshold])
            print('At treshold %1.3f, discard: '%threshold, ['%s(%d, %1.3f)'%(tag, self.numUsers[tag], self.thumbRatings.getRating(tag)) for tag in discard - saveDiscard])
            saveDiscard = saveDiscard | discard




def extractSurveyTags():
    """
    Build the file DIR_FILES_ROOT/survey/tags_survey
    Build a set of tags according to the default values in this module, sort by number of applications (desc.) and
    write to the file.
    This function differs from extractTags in the values of excludeActors and excludeDirectors
    NOTE: This function is called by the __main__ part of this module
    """
    #ts = tagset.TagSet(minUsers=MIN_USERS, minPercentile=MIN_PERCENTILE, excludeActors=True,
    #        excludeDirectors=True, excludeTagIfAllStopWords=EXCLUDE_TAG_IF_ALL_STOP_WORDS)
    ts = TagSet(minUsers=MIN_USERS, minPercentile=MIN_PERCENTILE, excludeActors=True,
            excludeDirectors=True, excludeTagIfAllStopWords=EXCLUDE_TAG_IF_ALL_STOP_WORDS)
    f = open(EXTRACT_FILE_SURVEY, 'w')
    dataSort = [(ts.numUsers[tag], tag) for tag in ts.tags]
    dataSort.sort()
    dataSort.reverse()
    f.write('\n'.join([tag for numUsers, tag in dataSort]))
    f.close()


def extractTags():
    """
    Build the file DIR_FILES_ROOT/survey/tags
    Build a set of tags according to the default values in this module, sort by number of applications (desc.) and
    write to the file.
    This function differs from extractSurveyTags in the values of excludeActors and excludeDirectors
    NOTE: This function is called by the __main__ part of this module
    """
    #ts = tagset.TagSet(minUsers=MIN_USERS, minPercentile=MIN_PERCENTILE, excludeActors=False,
    #        excludeDirectors=False, excludeTagIfAllStopWords=EXCLUDE_TAG_IF_ALL_STOP_WORDS)
    ts = TagSet(minUsers=MIN_USERS, minPercentile=MIN_PERCENTILE, excludeActors=False,
            excludeDirectors=False, excludeTagIfAllStopWords=EXCLUDE_TAG_IF_ALL_STOP_WORDS)
    f = open(EXTRACT_FILE, 'w')
    dataSort = [(ts.numUsers[tag], tag) for tag in ts.tags]
    dataSort.sort()
    dataSort.reverse()
    f.write('\n'.join([tag for numUsers, tag in dataSort]))
    f.close()


def getSurveyTagsFromFile():
    """
    :return: The set of tags from EXTRACT_FILE_SURVEY
    """
    f = open(EXTRACT_FILE_SURVEY, 'r')
    tags = set([line.strip() for line in f])
    return tags


def getTagsFromFile(file_with_tags=TAGS_SURVEY_BOOKS):
    """
    :return: The set of tags from EXTRACT_FILE
    """
    f = open(file_with_tags, 'r')
    tags = set([line.strip() for line in f])
    logger.info(f"Load tags from {file_with_tags}")
    return tags


if __name__ == '__main__':
    extractSurveyTags()
    extractTags()
