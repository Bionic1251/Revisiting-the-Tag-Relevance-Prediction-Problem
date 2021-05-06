y <- read.table(file="movielens-tagnav/r/out/y.csv", header=FALSE)$V1
y
surveyPredictions <- read.table(file="movielens-tagnav/r/out/surveyPredictions.csv", sep=",",fileEncoding = "utf-8")$V1

meanabserror <- function(actual, predicted){
  mean(abs(actual - predicted))
}

scale01 <- function(vec){
  (vec - 1.0) / 4
}
#print(scale01(2.5))
print(round(meanabserror(y, surveyPredictions),4))
print(round(meanabserror(scale01(y), scale01(surveyPredictions)),4))
