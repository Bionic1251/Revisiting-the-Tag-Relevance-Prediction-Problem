makeFeaturesDump <- function(){
  source("predict_relevance_application.R")
  arg1_train_relevance    <- "../../../data/processed/train_relevance.txt" 
  arg2_predict_relevance  <- "../../../data/processed/predict_relevance.txt"
  arg3_output_predictions <- "../../../data/predictions/relevance_predictions_books.txt"
  #arg4_data_survey_file   <- "../../../data/processed/data_survey_books.csv"
  arg4_data_survey_file   <- "../../../data/processed/features_r.csv"
  buildFeatures(arg1_train_relevance,
                arg2_predict_relevance, 
                arg3_output_predictions, 
                arg4_data_survey_file, 
                evalMode=FALSE) 
}

makeFeaturesDump()
