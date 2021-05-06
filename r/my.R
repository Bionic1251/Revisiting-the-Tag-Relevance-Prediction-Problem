test <- function(){
  source("movielens-tagnav/r/my_predict_relevance_application.R")
  arg1 <- "datasets/tagnav/temp/temp/r/data/train_relevance.txt"
  arg2 <- "datasets/tagnav/temp/temp/r/data/predict_relevance.txt"
  arg3 <- "datasets/tagnav/temp/temp/r/predictions/relevance_predictions_.txt"
  # predictTagRelevance(arg1, arg2, arg3, evalMode=FALSE)
  predictTagRelevance(arg1, arg2, arg3, evalMode=FALSE)
}
test()
