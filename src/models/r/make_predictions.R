modelPredictionsUsingFeaturesFile <- function(recalc=TRUE){
  source("my_predict_relevance_application.R")
	source("utils.R")
  trainFileBooks <- "../../../data/processed/train_test/train_books.csv"
  trainFileBooksAndMovies <- "../../../data/processed/train_test/train_books_and_movies.csv"
  testFile <- "../../../data/processed/train_test/test_books.csv"
  outputPredictionsBooks <- "../../../temp/r_predictions_b.dat"
  outputPredictionsBooksAndMovies <- "../../../temp/r_predictions_bm.dat"

  trainBooks <- read.table(file=trainFileBooks, header=TRUE, sep=",", quote = '"')
  trainBooksAndMovies <- read.table(file=trainFileBooksAndMovies, header=TRUE, sep=",", quote = '"')
  testDF <- read.table(file=testFile, header=TRUE, sep=",", quote = '"')

  if (recalc){
    trainPredictTest(trainBooks, testDF, outputPredictionsBooks)
    trainPredictTest(trainBooksAndMovies, testDF, outputPredictionsBooksAndMovies)
  } 

  y_true <- testDF$targets
  y_pred_b <- read.table(outputPredictionsBooks, header=FALSE)$V1
  y_pred_bm <- read.table(outputPredictionsBooksAndMovies, header=FALSE)$V1
  print(length(y_true))
  print(length(y_pred_b))
  print(length(y_pred_bm))
	mae_b <- meanabserror(y_pred_b, y_true)
	mae_bm <- meanabserror(y_pred_bm, y_true)
  print(paste("MAE Books:", mae_b))
  print(paste("MAE Books and Movies:", mae_bm))
}
modelPredictionsUsingFeaturesFile(recalc=FALSE)
