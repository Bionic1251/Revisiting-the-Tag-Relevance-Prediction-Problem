""" Based on load_imdb.py """
import time
import math
from loguru import logger
import gc
from src.data.ratings_sim import getSurveyTagsFromFile, getSurveyMoviesFromFile
from src.data.paths import *
from src.data.stopwords import stopwords
import re
import pickle as cPickle
from src.data.porter import PorterStemmer
from src.data.data import *
from src.data import utils
import json
from src.common import Input
from src.input_data import *


def make_imdb_obj(survey_tags=[],
                  survey_movies=[],
                  application_movies=[],
                  m_or_b=Input.BOOKS,
                  min_doc_rank=100):
    # Minimum number of documents a (non-tag) term must appear in to be included in the file.
    #surveyTags = getSurveyTagsFromFile(fpath=fpath_tags_survey)
    #surveyMovies = getSurveyMoviesFromFile(fpath=fpath_survey_movies_ids)
    #applicationMovies = getMoviesFromFile(fpath=fpath_all_movies_ids)
    all_movies = list(set(survey_movies)|set(application_movies))
    logger.info(f"All movies len = {len(all_movies)}")
    # First write out version of file where tags are stemmed
    c = IMDBCorpus(survey_tags)
    c.load(includeMovies=all_movies, minDocRank=min_doc_rank, DEBUG=True, m_or_b=m_or_b)
    c.normalize(betaSmooth=True)
    return c


def make_imdb_obj_nostem(survey_tags=[],
                         survey_movies=[],
                         application_movies=[],
                         m_or_b=Input.BOOKS,
                         min_doc_rank=100):
    all_movies = list(set(survey_movies)|set(application_movies))
    logger.info(f"All movies len = {len(all_movies)}")
    c = IMDBCorpus(survey_tags, stem=False)
    c.load(includeMovies=all_movies, minDocRank=min_doc_rank, DEBUG=True, m_or_b=m_or_b)
    c.normalize(betaSmooth=True)
    return c


def make_imdb_pickle(fpath_tags_survey=EXTRACT_FILE_SURVEY_MOVIES,
                     fpath_survey_movies_ids=EXTRACT_FILE_MOVIES,
                     fpath_all_movies_ids=FILE_MOVIES,
                     min_doc_rank=100,
                     out_pickle_imdb="imdb.pickle",
                     out_pickle_imdb_nostem="imdb_nostem.pickle"):
    # Minimum number of documents a (non-tag) term must appear in to be included in the file.
    surveyTags = getSurveyTagsFromFile(fpath=fpath_tags_survey)
    surveyMovies = getSurveyMoviesFromFile(fpath=fpath_survey_movies_ids)
    applicationMovies = getMoviesFromFile(fpath=fpath_all_movies_ids)
    allMovies = list(set(surveyMovies) | set(applicationMovies))

    # First write out version of file where tags are stemmed
    c = IMDBCorpus(surveyTags)
    c.load(includeMovies=allMovies, minDocRank=min_doc_rank, DEBUG=True, m_or_b=input_folder)
    c.normalize(betaSmooth=True)
    # cPickle.dump(c, open(FILE_IMDB_CORPUS, 'wb'))
    cPickle.dump(c, open(out_pickle_imdb, 'wb'))
    logger.info(f"Dump into {FILE_IMDB_CORPUS}")
    del c
    gc.collect()

    # Second write out version of file where tags are not stemmed
    c = IMDBCorpus(surveyTags, stem=False)
    c.load(includeMovies=allMovies, minDocRank=min_doc_rank, DEBUG=True, m_or_b=input_folder)
    c.normalize(betaSmooth=True)
    # cPickle.dump(c, open(FILE_IMDB_NOSTEM_CORPUS, 'wb'))
    cPickle.dump(c, open(out_pickle_imdb_nostem, 'wb'))
    logger.info(f"Dump into {FILE_IMDB_NOSTEM_CORPUS}")
    del c
    gc.collect()



def getMoviesFromFile(fpath=FILE_MOVIES):
    """
    :return the set of movies from FILE_MOVIES
    """
    f = open(fpath, 'r')
    movies = set([int(line.strip()) for line in f])
    return movies


class MovieDb:
    def __init__(self,
                 minRatings=50,
                 title=True,
                 avgRating=True,
                 popularity=False,
                 imdbId=False,
                 removeDate=False,
                 getReleaseDate=False):
        #mlcnx = sql_cnx.getMLConnection()
        #mlcursor = mlcnx.cursor()
        #sql = """
        #    select movieId, title, popularity, avgRating, imdbId, filmReleaseDate from movie_data where popularity >= %d
        #""" % minRatings
        #mlcursor.execute(sql)
        # movieData = mlcursor.fetchall()

        #movieData = df_movie_data[['item_id', 'title', 'popularity', 'avg_rating', 'imdb_id', 'filmReleaseDate']]
        movieData = df_movie_data[['item_id', 'title', 'avg_rating', 'imdb_id']]
        # movieData = movieData[movieData['popularity'] > minRatings].to_numpy().tolist()
        movieData = movieData.to_numpy().tolist()
        #self.movies = [movieId for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData]
        self.movies = [item_id for item_id, _, _, _ in movieData]
        if title:
            if removeDate:
                #self.titles = dict([(movieId, self.__removeDate(title)) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
                self.titles = dict([(item_id, self.__removeDate(title)) for item_id, title, _, _ in movieData])
            else:
                #self.titles = dict( [(movieId, title) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
                self.titles = dict( [(item_id, title) for item_id, title, avgRating, imdbId in movieData])
        if popularity:
            #self.popularity = dict( [(movieId, popularity) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
            self.popularity = dict( [(item_id, popularity) for item_id, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
        if avgRating:
            #self.avgRating = dict( [(movieId, avgRating) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
            self.avgRating = dict( [(item_id, avgRating) for item_id, title, avgRating, imdbId in movieData])
        if imdbId:
            self.imdbIds = dict( [(item_id, imdbId) for item_id, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
        if getReleaseDate:
            self.releaseDates = dict( [(item_id, filmReleaseDate) for item_id, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])

    def hasMovie(self, movieId):
        return movieId in self.titles

    def getTitle(self, movieId):
        return self.titles[movieId]

    def getPopularity(self, movieId):
        return self.popularity[movieId]

    def getAvgRating(self, movieId):
        return self.avgRating[movieId]

    def getImdbId(self, movieId):
        return self.imdbIds[movieId]

    def getReleaseDate(self, movieId):
        return self.releaseDates[movieId]

    def getTitles(self):
        return self.titles.values()

    def __removeDate(self, title):
        parenBegin = title.find('(')
        if parenBegin >= 0:
            return title[0:parenBegin]
        else:
            return title


ARTICLES = re.compile(r"\ba\b|\ban\b|\bthe\b")


def getStopWords():
    return stopwords
    #return set([line.strip() for line in open(FILE_STOPWORDS, 'r')])


class Corpus:
    """
    This class represents a set of documents, each as a bag of words (or multi-word tokens)

    Instance variables:
        minDocs:  the minimum number of documents a word must appear in to be retained
        removeStopWords:  if True, remove the stop words
        stem:  if True, stem words
        DEBUG:  if True, print verbosely
        maxProcess:  the max number of documents to process
        includeDocs:  a list of documents to include, even if docCount >= maxProcess
        stopWords:  a set of stopwords (retrieved from the stopwords file)
        stemmer:  the PorterStemmer object for stemming
        wordDocCount:  number of docs each word appears in
        docVectors:  dictionary of vectors of word counts for each document, indexed by word
        docCount:  the number of documents that have been processed
        wordToIndex:  a dictionary from string word to the wordIndex at the time of processing
        indexToWord:  a dictionary from index (wordIndex at the time the word was processed) to the string word
        wordIndex:  the current count of the number of words (words are indexed with this increasing as they are processed)
        smoothingWords:  an int to add to the count of word occurrences when normalizing word frequency
        freq:  True if docVectors was normalized by word frequencies
        betaSmooth:  True if docVectors was normalized by beta smoothing
        tfidf:  True if docVectors was transformed by term frequency-inverse document frequency
        tags:  a list of preprocessed tags (including spaces)
        multiwordTagsUnformatted:  a list of members of the constructor param list `tags` that contain a space (no preprocessing done)
        preprocessedTags:  a list of the (non-empty-string) results of preprocessing the input tags and removing spaces
        multiwordDict:  a dictionary of preprocessed tag to the preprocessed tag after removing spaces
        multiwordPattern:  a regex pattern for matching the tags in multiwordDict
        movieDb:  an ml.moviedb.MovieDb object (with dates removed from titles)
        excludeTitles:  a list, the result of preprocessing a list of movie titles
            * Only consider titles of movies with enough ratings
            * Preprocess the titles by lowercasing and removing punctuation, articles, and extra spaces
            * Do not exclude titles that are only a single word after preprocessing
            * Do not exclude titles if the results of preprocessing them matches any tags
    """

    def __init__(self, tags, minDocs=20, removeStopWords=True, stem=True, DEBUG=False, maxProcess=0, includeDocs=[]):
        """
        :param tags: the unprocessed list of tags to process
        :param minDocs: the minimum number of documents a word must appear in to be retained
        :param removeStopWords: if True, remove the stop words during preprocessing
        :param stem: if True, stem words during preprocessing
        :param DEBUG: if True, print verbosely
        :param maxProcess: the max number of documents to process
        :param includeDocs: a list of documents to include, even if docCount >= maxProcess
        """
        if DEBUG:
            print('Initializing corpus' + time.strftime('%x %X') + '\n')

        self.minDocs = minDocs
        self.removeStopWords = removeStopWords
        self.stem = stem

        self.DEBUG = DEBUG
        self.maxProcess = maxProcess
        self.includeDocs = includeDocs

        if removeStopWords:
            self.stopWords = getStopWords()
        if stem:
            self.stemmer = PorterStemmer()

        self.wordDocCount = {}  # Number of docs each word appears in
        self.docVectors = {}  # List of vectors of word counts for each document, indexed by word
        self.docCount = 0
        self.wordToIndex = {}
        self.indexToWord = {}
        self.wordIndex = 0
        self.smoothingWords = 0

        self.freq = False
        self.betaSmooth = False
        self.tfidf = False

        self.tags = []
        self.multiwordTagsUnformatted = set([])
        self.preprocessedTags = {}

        # Preprocess and format input tags, adding to self.multiwordTagsUnformatted, self.tags, and self.preprocessedTags
        for tag in tags:
            if tag.find(' ') > 0:
                self.multiwordTagsUnformatted.add(tag)
            formattedTag = self.preprocessText(tag, groupTags=False)
            if formattedTag != '':
                self.tags.append(formattedTag)
            self.preprocessedTags[tag] = formattedTag.replace(' ', '')
        self.multiwordDict = dict([(tag, tag.replace(' ', '')) for tag in self.tags if tag.find(' ') >= 0])
        self.multiwordPattern = re.compile('|'.join(map(re.escape, self.multiwordDict)))

        self.movieDb = MovieDb(removeDate=True)

        excludeTitles = [title.lower() for title in MovieDb(removeDate=True, minRatings=1000).getTitles()]
        excludeTitles = map(self.removeExtraSpaces,
                            (map(self.removeArticles, map(self.removePunctuation, excludeTitles))))
        excludeTitles = [title for title in excludeTitles if
                         len(title.split()) > 1]  # Do not exclude single-word titles
        doNotExcludeTheseTags = set(
            map(self.removeExtraSpaces, (map(self.removeArticles, map(self.removePunctuation, tags)))))
        self.excludeTitles = [title for title in excludeTitles if title not in doNotExcludeTheseTags]


    def addUpdateDoc(self, name, text, m_or_b=Input.MOVIES):
        """
        Add or update a document
        :param name: the name of the document
        :param text: the text of the document
        """
        self.docCount += 1
        if self.docCount > self.maxProcess and self.maxProcess != 0 and name not in self.includeDocs:
            return
        if self.DEBUG:
            if self.docCount % 500 == 0:
                print('%d docs processed in loop 1 ' % self.docCount + ' num unique terms:%s ' % len(
                    self.wordDocCount.keys()) + time.strftime('%x %X') + '\n')

        if m_or_b is Input.MOVIES:
            # Ignore the title of this document (remove it from text)
            text = self.preprocessText(text, removeTitle=self.movieDb.getTitle(name))


        # Track words for this document in docVectors
        docVector = self.docVectors.get(name, {})
        for word in text.split():
            if word not in self.wordToIndex:
                # New word found, index it
                self.wordToIndex[word] = self.wordIndex
                self.indexToWord[self.wordIndex] = word
                w_index = self.wordIndex
                self.wordIndex += 1
            else:
                w_index = self.wordToIndex[word]
            if w_index not in docVector:
                docVector[w_index] = 1
                if w_index not in self.wordDocCount:
                    self.wordDocCount[w_index] = 1
                else:
                    self.wordDocCount[w_index] += 1
            else:
                docVector[w_index] += 1
        self.docVectors[name] = docVector

    def preprocessText(self, text, groupTags=True, removeTitle=None):
        """
        Prepare the input text for processing by
            * lowercasing
            * removing punctuation
            * remove article words
            * searching for/removing the title if non-None
            * if self.removeStopWords, then remove stop words
            * if self.stem, then stem the words in the text
            * if groupTags, then do self.groupMultiWordTags
        :param text: the string to preprocess
        :param groupTags: boolean flag indicating whether to
        :param removeTitle: the string title of the movie to remove, or None if the title should not be removed
        :return: the string result of preprocessing the text
        """
        try:
            return self.preprocessedTags[text]
        except LookupError:
            pass

        if text == 'harsh':
            return 'harsh'

        text = text.lower()
        text = self.removePunctuation(text)
        text = self.removeArticles(text)

        if removeTitle:
            # First, remove title of movie being evaluated
            # Match based on any prefix of title at least 2 words long
            # If prefix of title contains tags with multiple words, replace title with tag
            removeTitle = removeTitle.lower()
            removeTitle = self.removePunctuation(removeTitle)
            removeTitle = self.removeArticles(removeTitle)
            titleWords = removeTitle.split()
            for numWords in reversed(range(2, len(titleWords) + 1)):
                partialTitle = ' '.join(titleWords[0:numWords])
                # print partialTitle
                for numWordsTag in reversed(range(2, numWords + 1)):
                    candidateTag = ' '.join(titleWords[0:numWordsTag])
                    if candidateTag in self.multiwordTagsUnformatted:
                        matchingTag = candidateTag
                        # print partialTitle, '>', matchingTag
                        break
                    else:
                        matchingTag = ''
                text = re.sub('\\b' + partialTitle + '\\b', matchingTag, text)
            # Second, match on popular titles
            # Look at whole title (not prefix) to avoid false positives
            for otherTitle in self.excludeTitles:
                if text.find(otherTitle) >= 0:
                    text = re.sub('\\b' + otherTitle + '\\b', '', text)

        if self.removeStopWords:
            text = self.doRemoveStop(text, self.stopWords)

        if self.stem:
            text = self.doStem(text, self.stemmer)

        if groupTags:
            text = self.groupMultiWordTags(text)

        return text

    def groupMultiWordTags(self, text):
        """
        :return: the string result of grouping any occurrences of multi-word tags
        """
        def floc(x):
            return self.multiwordPattern.sub(lambda match: self.multiwordDict[match.group(0)], x)

        #return self.multiwordPattern.sub(lambda match: self.multiwordDict[match.group(0)], text)
        return self.multiwordPattern.sub(lambda match: self.multiwordDict.get(match.group(0)), text)

    def removeArticles(self, text):
        """
        :return: the string result of removing article words (see the ARTICLES regex pattern)
        """
        return ARTICLES.sub('', text)

    def removeExtraSpaces(self, text):
        """
        :return: the string result of condensing spaces (ie: if text was "a   b", return "a b"
        """
        return ' '.join(text.split())

    def updateDocVectors(self):
        """
        Prune words (based on minDocs), and update the indices
        """
        if self.DEBUG:
            print('Finished, point A: num unique terms:%s ' % len(self.wordDocCount.keys()) + time.strftime(
                '%x %X') + '\n')

        # Handle multi-word tags
        groupedTags = set([self.multiwordDict.get(tag, tag) for tag in self.tags])
        includeWords = set([self.wordToIndex[word] for word in groupedTags if word in self.wordToIndex])

        # Remove words that don't appear in at least minDoc documents, and re-index the words
        retainedWords = [w_index for w_index, count in self.wordDocCount.items() if
                         count > self.minDocs or w_index in includeWords]
        retainedWords.sort()
        oldToNewIndex = dict([(w_index, index) for index, w_index in enumerate(retainedWords)])

        # Update self.wordDocCount
        self.wordDocCount = self.translateKeys(self.wordDocCount, oldToNewIndex)

        if self.DEBUG:
            print('Finished, point A: num unique terms:%s ' % len(self.wordDocCount.keys()) + time.strftime(
                '%x %X') + '\n')

        # Update docVectors
        for name, docVector in self.docVectors.items():
            self.docVectors[name] = self.translateKeys(docVector, oldToNewIndex)

        # Update word indices
        self.indexToWord = self.translateKeys(self.indexToWord, oldToNewIndex)
        self.wordToIndex = dict([(word, index) for index, word in self.indexToWord.items()])

        # Update counts of words after pruning
        self.wordCounts = {}
        for name, docVector in self.docVectors.items():
            self.wordCounts[name] = sum(docVector.values())

        if self.DEBUG:
            print('num terms after ' + str(self.getNumTerms2()))
            print('Done loading corpus ' + time.strftime('%x %X') + '\n')

        self.indexToDocName = dict([(index, name) for index, name in enumerate(sorted(self.docVectors.keys()))])
        self.docNameToIndex = dict([(name, index) for index, name in self.indexToDocName.items()])
        #        if self.stem:
        #            del self.stemmer
        #            self.stemmer = porter.PorterStemmer() # Clear out dictionary of saved words
        self.computeBackgroundWordFreqs()

    def translateKeys(self, d, oldToNewIndex):
        """
        :param d: the dictionary (from index to some value) to translate to a new set of indices
        :param oldToNewIndex: a dictionary of old index to new index
        :return: the result of translating d to use the new indices
        NOTE: this prunes the dictionary of indices that have no translation
        """
        return dict([(oldToNewIndex[k], v) for k, v in d.items() if k in oldToNewIndex])

    def setMinDocs(self, minDocs):
        """
        Set self.minDocs, and reindex/prune structures
        """
        self.minDocs = minDocs
        self.updateDocVectors()

    def setMinDocRank(self, rank):
        """
        Determine the number of documents containing the word ranked at rank, then call self.setMinDocs with that number

        Mine: using Python 2.7
        a=[('a',100), ('b', 12), ('e', 8)]
        b=zip(*a)
        print(b)
        > [('a', 'b', 'e'), (100, 12, 8)]
        c=b[1]
        print(c)
        > (100, 12, 8)


        print(list(zip(*a)[1]))
        > (100, 12, 8)

        """
        # docCounts = list(zip(*self.getTermDocCount())[1])
        # my:
        docCounts = [e[1] for e in self.getTermDocCount()]
        docCounts.sort()
        docCounts.reverse()
        try:
            self.setMinDocs(docCounts[rank])
        except IndexError:
            logger.info(f"Rank {rank} is not present, setting to 1")
            self.setMinDocs(docCounts[1])


    def normalize(self, freq=False, betaSmooth=False, tfidf=False):
        """
        Perform normalization(s) on docVectors
        :param freq: if True, normalize by word frequency
        :param betaSmooth: if True, normalize by beta smoothing
        :param tfidf: if True, transform by term frequency-inverse document frequency
        NOTE: freq and beatSmooth cannot both be True
        """
        if self.freq or self.betaSmooth or self.tfidf:
            raise Exception('Data has already been normalized')
        if freq and betaSmooth:
            raise Exception('Conflicting normalization settings.')

        self.tfidf = tfidf
        self.freq = freq
        self.betaSmooth = betaSmooth

        if self.betaSmooth:
            self.computeAlphaBeta()

        for docVector in self.docVectors.values():
            if self.freq == True:
                self.doFreq(docVector)
            elif self.betaSmooth == True:
                self.doBetaSmooth(docVector)
            if self.tfidf == True:
                self.doTfidf(docVector)

        self.computeTagFreqMeanStdDev()

    def doFreq(self, docVector):
        """
        Normalize docVector by word frequency
        """
        numTerms = sum(docVector.values()) + self.smoothingWords
        for w_index, count in docVector.items():
            docVector[w_index] = float(count) / numTerms

    def doBetaSmooth(self, docVector):
        """
        Normalize docVector by beta smoothing
        """
        numTerms = sum(docVector.values())
        for w_index, count in docVector.items():
            alpha = self.alphas[w_index]
            beta = self.betas[w_index]
            docVector[w_index] = (float(count) + alpha) / (numTerms + alpha + beta)

    def getBetaSmoothedZeroFreq(self, word, movieId):
        # This does not check that the frequency is actually zero
        if not self.betaSmooth:
            return 0
        word = self.preprocessText(word)
        try:
            wordIndex = self.wordToIndex[word]
        except LookupError:
            return 0
        alpha = self.alphas[wordIndex]
        beta = self.betas[wordIndex]
        numTerms = self.numTerms[movieId]
        return alpha / (numTerms + alpha + beta)

    def doTfidf(self, docVector):
        """
        Transform docVector by term frequency-inverse document frequency
        """
        numDocs = len(self.docVectors)
        for w_index, val in docVector.items():
            docVector[w_index] = val * math.log(float(numDocs) / self.wordDocCount[w_index])

    def computeAlphaBeta(self, DEBUG=False):
        wordFreqs = dict([(wordIndex, []) for wordIndex in self.indexToWord])
        self.numTerms = {}
        for name, docVector in self.docVectors.items():
            numTerms = sum(docVector.values())
            self.numTerms[name] = numTerms
            for wordIndex, count in docVector.items():
                wordFreqs[wordIndex].append(float(count) / numTerms)

        numDocs = len(self.docVectors)
        self.alphas = {}
        self.betas = {}
        for wordIndex, freqs in wordFreqs.items():
            numZeroFreqs = numDocs - len(freqs)
            freqVector = freqs[:]
            freqVector.extend([0.0] * numZeroFreqs)
            m = utils.mean(freqVector)
            v = utils.variance(freqVector)
            try:
                self.alphas[wordIndex] = m * ((m * (1 - m) / v) - 1)
            except ZeroDivisionError:
                self.alphas[wordIndex] =  0.0
                logger.info(f"ZeroDivisionError: v ={v}")
            try:
                self.betas[wordIndex] = (1 - m) * ((m * (1 - m) / v) - 1)
            except ZeroDivisionError:
                self.betas[wordIndex] = 0.0
                logger.info(f"ZeroDivisionError: v ={v}")
        if DEBUG:
            for wordIndex in self.alphas:
                alpha = self.alphas[wordIndex]
                beta = self.betas[wordIndex]
                print(self.indexToWord[wordIndex], 'alpha=', alpha, 'beta=', beta, 'alpha/(alpha + beta)',
                      alpha / (alpha + beta))

    def computeTagFreqMeanStdDev(self):
        self.tagFreqMean = {}
        self.tagFreqStdDev = {}
        for tag in self.preprocessedTags.values():
            if tag not in self.wordToIndex:  # tag never appears in the corpus
                self.tagFreqMean[tag] = 0
                self.tagFreqStdDev[tag] = 0
                continue
            wordIndex = self.wordToIndex[tag]
            vals = []
            if self.betaSmooth:
                alpha = self.alphas[wordIndex]
                beta = self.betas[wordIndex]
            for movieId, docVector in self.docVectors.items():
                try:
                    val = docVector[wordIndex]
                except:
                    if self.betaSmooth:
                        val = alpha / (self.numTerms[movieId] + alpha + beta)
                    else:
                        val = 0
                vals.append(val)
            self.tagFreqMean[tag] = utils.mean(vals)
            self.tagFreqStdDev[tag] = math.sqrt(utils.variance(vals))

    def getTagFreqMeanStdDev(self, tag):
        tag = self.preprocessText(tag)
        try:
            freq_mean = self.tagFreqMean[tag]
            freq_std = self.tagFreqStdDev[tag]
            return freq_mean, freq_std
        except KeyError as e:
            #logger.info(f"TODO: tag={tag} not found")
            return 0, 0


    def computeBackgroundWordFreqs(self):
        self.wordFreqs = dict([(wordIndex, 0) for wordIndex in self.indexToWord.keys()])
        for docVector in self.docVectors.values():
            for wordIndex, count in docVector.items():
                self.wordFreqs[wordIndex] += count
        totalWordCount = float(sum(self.wordFreqs.values()))
        for wordIndex, count in self.wordFreqs.items():
            self.wordFreqs[wordIndex] = count / totalWordCount

    def getBackgroundWordFreq(self, text):
        word = self.preprocessText(text)
        try:
            wordIndex = self.wordToIndex[word]
        except LookupError:
            return 0
        return self.wordFreqs[wordIndex]

    def getDocVector(self, name):
        """
        Get the docVector corresponding to name, or None if name is not found
        """
        return self.docVectors.get(name, None)

    def getMinWordFreq(self, name, text):
        text = self.preprocessText(text)
        docVector = self.getDocVector(name)
        if docVector == None:
            return 0
        wordFreq = []
        for word in text.split():
            if word not in self.wordToIndex:
                return 0
            wordFreq.append(docVector.get(self.wordToIndex[word], 0))
        if len(wordFreq) == 0:
            return 0
        return min(wordFreq)

    def getWordFreq(self, name, text):
        text = self.preprocessText(text)
        docVector = self.getDocVector(name)
        if docVector == None:
            return None
        return docVector.get(self.wordToIndex.get(text, -1), 0)

    def getCountsForTag(self, unformattedTag):
        tag = self.preprocessText(unformattedTag)
        if tag not in self.wordToIndex:
            return dict([(name, 0) for name, docVector in self.docVectors.items()])

        tagIndex = self.wordToIndex[tag]
        return dict([(name, docVector.get(tagIndex, 0)) for name, docVector in self.docVectors.items()])

    def getCountsForTagPretty(self, unformattedTag):
        countMovieId = [(count, movieId) for movieId, count in self.getCountsForTag(unformattedTag).items()]
        countMovieId.sort()
        countMovieId.reverse()
        for count, movieId in countMovieId:
            print('%1.5f\t%d\%s' % (count, movieId, self.movieDb.getTitle(movieId)))

    def getTotalFreq(self, name):
        docVector = self.getDocVector(name)
        if docVector == None:
            return None
        return sum(docVector.values())

    def densify(self, docVector):
        denseVector = [0] * len(self.indexToWord)
        for w_index, val in docVector.items():
            denseVector[w_index] = val
        return denseVector

    def makeDocVectorFromText(self, text):  # Assumes each word appears just once
        text = self.preprocessText(text)
        docVector = {}
        for word in text.split():
            docVector[self.wordToIndex[word]] = 1
        return docVector

    def getDistinctWords(self):
        """
        :return: a sorted list of the indexed words
        """
        return sorted(self.wordToIndex.keys())

    def getDocVectorPretty(self, name):
        wordCounts = [(count, self.indexToWord[index]) for index, count in self.getDocVector(name).items()]
        wordCounts.sort()
        wordCounts.reverse()
        return [(word, count) for count, word in wordCounts]

    def removePunctuation(self, text, keepCharacter=''):
        """
        :param text: the text to prune
        :param keepCharacter: a string of punctuation characters to keep
        :return: the string result of removing punctuation from the text
        """
        return re.sub(r"[^A-Za-z0-9 " + keepCharacter + "]", " ", text)

    def doRemoveStop(self, text, stopWords):
        """
        :param text: the text to prune
        :param stopWords: a list of words to be considered as stop words
        :return: the string result of removing all stop words and punctuation from the text
        """
        return ' '.join([word for word in self.removePunctuation(text, "'").split() if word not in stopWords])

    def doStem(self, text, stemmer):
        """
        :param text: the text to stem
        :param stemmer: the stemmer object for stemming
        :return: the string result after all words have been stemmed
        """
        return ' '.join([stemmer.stem(word, adjustCase=False) for word in text.split()])

    def getNumDocs(self, word):
        """
        :return: The number of documents word appears in after preprocessing
        """
        word = self.preprocessText(word)
        if word not in self.wordToIndex:
            return 0
        return self.wordDocCount.get(self.wordToIndex[word], 0)

    def getNumTerms(self):
        """
        :return: the number of words indexed
        """
        return len(self.wordToIndex.keys())

    def getTermDocCount(self):
        """
        :return: a list of (word, count) tuples, sorted by decreasing counts
        """
        countsTerms = [(count, w_index) for w_index, count in self.wordDocCount.items()]
        countsTerms.sort()
        countsTerms.reverse()
        return [(self.indexToWord[w_index], count) for count, w_index in countsTerms]

    def getName(self):
        """
        :return: the string name of this corpus (must be overridden by sub-classes)
        """
        pass


IMDB_CORPUS_NAME = "IMDB"

class IMDBCorpus(Corpus):
    """
    This class represents the corpus of IMDb movie reviews.
        A movie is a "document", and review text is processed into words for that movie/document
    """

    def load(self, includeMovies, minDocRank, DEBUG=False, m_or_b=Input.MOVIES):
        """
        Load the data from previous scrapes (from imdb_review table)
        :param includeMovies: a list of movies to consider
        :param minDocRank:
        """
        if DEBUG:
            print('Starting to load imdb data' + time.strftime('%x %X')+ '\n')

        #sql = """ select heading, body from imdb_review where movieId=%s """
        #sql = """ select heading, body from imdb_corpus where movieId=%s """

        n_movies = len(includeMovies)
        reviews_obj = read_reviews_json()
        missing_ids = []
        for i, movieId in enumerate(includeMovies):
            if DEBUG:
                if (i + 1) % 100 == 0:
                    logger.info(f"Proceesed {100.0*i/n_movies} % out of {n_movies} movies")
            reviews = get_text_reviews_from_json_obj(reviews_obj, movieId)
            if reviews is None:
                missing_ids.append(movieId)
                # continue
                reviews = ''
            self.addUpdateDoc(movieId, reviews, m_or_b=m_or_b)

        # ???
        self.setMinDocRank(minDocRank)

        for _, vec in self.docVectors.items():
            # assert len(vec) > 0, "Empty doc vectors: set minDocRank to lower value"
            if len(vec) <= 0:
                logger.info("Empty doc vectors: set minDocRank to lower value")

        missing_ids = list(set(missing_ids))
        save_path = os.path.join(TEMP_DIR, 'missing_review_ids.json')
        with open(save_path, 'w') as f:
            json.dump({"ids": missing_ids}, f)
        logger.info(f"Save missing in reviews IDs into {save_path}")


    def load_old(self, includeMovies, minDocRank, DEBUG=False):
        # TODO (kyle): determine what minDocRank does
        """
        Load the data from previous scrapes
        :param includeMovies: a list of movies to consider
        :param minDocRank:
        """
        if DEBUG:
            print('Starting to load imdb data' + time.strftime('%x %X')+ '\n')

        # Get the names of the successful previously-scraped files from the DB
        tncnx = sql_cnx.getIMDBConnection()
        tncursor = tncnx.cursor()
        sql = "SELECT distinct fileName from %s where fileName != '' and status = 's' order by fileName" % TABLE_IMDB_EXTRACT
        tncursor.execute(sql)
        fileNames = [row[0] for row in tncursor.fetchall()]

        # Process successful previously-scraped files
        movieId=0
        numMoviesProcessed=0
        reviewText = ''
        saveFileName = ''
        fileName = ''
        for fileName in fileNames:
            f = open(os.path.join(DIR_IMDB_SCRAPE, fileName),'r')
            for line in f:
                code = line[line.find('<')+1 : line.find('>')]
                data = line[line.find('>')+1:].lower()
                if code=='imdblog_movie':
                    # Found the start of a new movie's reviews
                    if movieId != 0:
                        # This is not the first movie, so a previous movie just ended; process it
                        self.processMovie(movieId, reviewText, saveFileName, tncursor, includeMovies)
                        if DEBUG:
                            numMoviesProcessed +=1
                            if numMoviesProcessed % 500 == 0:
                                print('Processed %d movies '%numMoviesProcessed + time.strftime('%x %X')+ '\n')
                                sys.stdout.flush()
                    movieId=int(data)
                    reviewText = ''
                    saveFileName = fileName
                elif code=='imdblog_review_text' or code=='imdblog_review_heading' :
                    # Accumulate review info for the current movie
                    reviewText += ' ' + data

        # Process the final movie
        self.processMovie(movieId, reviewText, fileName, tncursor, includeMovies)

        self.setMinDocRank(minDocRank)


    def processMovie(self, movieId, reviewText, fileName, tncursor, includeMovies):
        """
        :param movieId: the movieId to process
        :param reviewText: the reviewText info from the file
        :param fileName: the name of the file the movie was found in
        :param tncursor: the MySQLdb cursor currently being used
        :param includeMovies: a list of movies to consider
        """
        if includeMovies and movieId not in includeMovies:
            return

        # Determine the status from the database
        sql = """ SELECT status from %s where movieId = %d and fileName = "%s" """ % (TABLE_IMDB_EXTRACT, movieId, fileName)
        tncursor.execute(sql)
        rows = tncursor.fetchall()
        if len(rows) <= 0:
            # Did not find this entry in the database
            return
        status= rows[0][0]

        # Update the info if the status is 's'
        reviewText = reviewText.replace("<br>", "")
        if status == 's':
            self.addUpdateDoc(movieId, reviewText)


    def getName(self):
        """
        Specify what kind of corpus
        """
        return IMDB_CORPUS_NAME


    def write(self,fileName='imdb'):
        """
        Dump this object to a file using pickle
        """
        full_path = os.path.join(DIR_IMDB_SCRAPE, fileName + '.pickle')
        cPickle.dump(self, open(full_path, 'wb'))

