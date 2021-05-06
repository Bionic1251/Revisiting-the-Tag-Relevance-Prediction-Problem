
runTenFolds <- function(){
  train0 = "libdat/data/10_folds/train0.csv"
  train1 = "libdat/data/10_folds/train1.csv"
  train2 = "libdat/data/10_folds/train2.csv"
  train3 = "libdat/data/10_folds/train3.csv"
  train4 = "libdat/data/10_folds/train4.csv"
  train5 = "libdat/data/10_folds/train5.csv"
  train6 = "libdat/data/10_folds/train6.csv"
  train7 = "libdat/data/10_folds/train7.csv"
  train8 = "libdat/data/10_folds/train8.csv"
  train9 = "libdat/data/10_folds/train9.csv"

  test0 = "libdat/data/10_folds/test0.csv"
  test1 = "libdat/data/10_folds/test1.csv"
  test2 = "libdat/data/10_folds/test2.csv"
  test3 = "libdat/data/10_folds/test3.csv"
  test4 = "libdat/data/10_folds/test4.csv"
  test5 = "libdat/data/10_folds/test5.csv"
  test6 = "libdat/data/10_folds/test6.csv"
  test7 = "libdat/data/10_folds/test7.csv"
  test8 = "libdat/data/10_folds/test8.csv"
  test9 = "libdat/data/10_folds/test9.csv"

  predictions0 = "libdat/temp/r_predictions_fold_0.txt"
  predictions1 = "libdat/temp/r_predictions_fold_1.txt"
  predictions2 = "libdat/temp/r_predictions_fold_2.txt"
  predictions3 = "libdat/temp/r_predictions_fold_3.txt"
  predictions4 = "libdat/temp/r_predictions_fold_4.txt"
  predictions5 = "libdat/temp/r_predictions_fold_5.txt"
  predictions6 = "libdat/temp/r_predictions_fold_6.txt"
  predictions7 = "libdat/temp/r_predictions_fold_7.txt"
  predictions8 = "libdat/temp/r_predictions_fold_8.txt"
  predictions9 = "libdat/temp/r_predictions_fold_9.txt"

  source("movielens-tagnav/r/my_predict_relevance_application.R")


  trainPrecictTest(train0, test0, predictions0)
  trainPrecictTest(train1, test1, predictions1)
  trainPrecictTest(train2, test2, predictions2)
  trainPrecictTest(train3, test3, predictions3)
  trainPrecictTest(train4, test4, predictions4)
  trainPrecictTest(train5, test5, predictions5)
  trainPrecictTest(train6, test6, predictions6)
  trainPrecictTest(train7, test7, predictions7)
  trainPrecictTest(train8, test8, predictions8)
  trainPrecictTest(train9, test9, predictions9)
}
runTenFolds()
