write.table(dataSurvey, file="/home/ms314/projects/movielens-tagnav/r/out/dataSurvey.csv", sep=",", col.names = TRUE, row.names = FALSE, fileEncoding = "utf-8")

write.table(y, file="/home/ms314/projects/movielens-tagnav/r/out/y.csv", sep=",", col.names = FALSE, row.names = FALSE, fileEncoding = "utf-8")

write.table(surveyPredictions, file="/home/ms314/projects/movielens-tagnav/r/out/surveyPredictions.csv", sep=",", col.names = FALSE, row.names = FALSE, fileEncoding = "utf-8")