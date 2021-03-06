runTenFolds <- function(){
  levelup = "../../../"

  train0 = paste0(levelup, "data/processed/10folds/train0.csv")
  train1 = paste0(levelup, "data/processed/10folds/train1.csv")
  train2 = paste0(levelup, "data/processed/10folds/train2.csv")
  train3 = paste0(levelup, "data/processed/10folds/train3.csv")
  train4 = paste0(levelup, "data/processed/10folds/train4.csv")
  train5 = paste0(levelup, "data/processed/10folds/train5.csv")
  train6 = paste0(levelup, "data/processed/10folds/train6.csv")
  train7 = paste0(levelup, "data/processed/10folds/train7.csv")
  train8 = paste0(levelup, "data/processed/10folds/train8.csv")
  train9 = paste0(levelup, "data/processed/10folds/train9.csv")

  test0 = paste0(levelup, "data/processed/10folds/test0.csv")
  test1 = paste0(levelup, "data/processed/10folds/test1.csv")
  test2 = paste0(levelup, "data/processed/10folds/test2.csv")
  test3 = paste0(levelup, "data/processed/10folds/test3.csv")
  test4 = paste0(levelup, "data/processed/10folds/test4.csv")
  test5 = paste0(levelup, "data/processed/10folds/test5.csv")
  test6 = paste0(levelup, "data/processed/10folds/test6.csv")
  test7 = paste0(levelup, "data/processed/10folds/test7.csv")
  test8 = paste0(levelup, "data/processed/10folds/test8.csv")
  test9 = paste0(levelup, "data/processed/10folds/test9.csv")

  predictions0 = paste0(levelup, "data/predictions/r_predictions_fold_0.txt")
  predictions1 = paste0(levelup, "data/predictions/r_predictions_fold_1.txt")
  predictions2 = paste0(levelup, "data/predictions/r_predictions_fold_2.txt")
  predictions3 = paste0(levelup, "data/predictions/r_predictions_fold_3.txt")
  predictions4 = paste0(levelup, "data/predictions/r_predictions_fold_4.txt")
  predictions5 = paste0(levelup, "data/predictions/r_predictions_fold_5.txt")
  predictions6 = paste0(levelup, "data/predictions/r_predictions_fold_6.txt")
  predictions7 = paste0(levelup, "data/predictions/r_predictions_fold_7.txt")
  predictions8 = paste0(levelup, "data/predictions/r_predictions_fold_8.txt")
  predictions9 = paste0(levelup, "data/predictions/r_predictions_fold_9.txt")

  source("my_predict_relevance_application.R")

  trainPredictTest(train0, test0, predictions0)
  trainPredictTest(train1, test1, predictions1)
  trainPredictTest(train2, test2, predictions2)
  trainPredictTest(train3, test3, predictions3)
  trainPredictTest(train4, test4, predictions4)
  trainPredictTest(train5, test5, predictions5)
  trainPredictTest(train6, test6, predictions6)
  trainPredictTest(train7, test7, predictions7)
  trainPredictTest(train8, test8, predictions8)
  trainPredictTest(train9, test9, predictions9)
}
runTenFolds()
