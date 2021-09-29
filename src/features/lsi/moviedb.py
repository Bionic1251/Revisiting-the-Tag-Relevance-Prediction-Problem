#import ml.sql_connections as sql_cnx


class MovieDb():

    def __init__(self, minRatings=50, title=True, avgRating=True, popularity=True, imdbId=False, removeDate=False, getReleaseDate=False):
        mlcnx = sql_cnx.getMLConnection()
        mlcursor = mlcnx.cursor()
        sql = """
            select movieId, title, popularity, avgRating, imdbId, filmReleaseDate from movie_data where popularity >= %d
        """%minRatings
        mlcursor.execute(sql)
        movieData = mlcursor.fetchall()
        self.movies = [movieId for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData]
        if title:
            if removeDate:
                self.titles = dict([(movieId, self.__removeDate(title)) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
            else:
                self.titles = dict([(movieId, title) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
        if popularity:
            self.popularity = dict([(movieId, popularity) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
        if avgRating:
            self.avgRating = dict([(movieId, avgRating) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
        if imdbId:
            self.imdbIds = dict([(movieId, imdbId) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])
        if getReleaseDate:
            self.releaseDates = dict([(movieId, filmReleaseDate) for movieId, title, popularity, avgRating, imdbId, filmReleaseDate in movieData])

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
