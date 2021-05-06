write.table(dataSurvey, file="movielens-tagnav/r/out/dataSurvey.csv", sep=",", col.names = TRUE, row.names = FALSE, fileEncoding = "utf-8")

write.table(y, file="movielens-tagnav/r/out/y.csv", sep=",", col.names = FALSE, row.names = FALSE, fileEncoding = "utf-8")

write.table(surveyPredictions, file="movielens-tagnav/r/out/surveyPredictions.csv", sep=",", col.names = FALSE, row.names = FALSE, fileEncoding = "utf-8")
