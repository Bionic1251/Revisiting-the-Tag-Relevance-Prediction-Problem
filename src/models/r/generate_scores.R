generateScores <- function(trainDataFrame, 
                             testDataFrame, 
                             savePredictionsPath) {
	library(lme4)
	source("models.R")
	source("dataset.R")
	source("tagfit.R")
	source("utils.R")
	source("crossval.R")
	source("step_regression.R")

  dataSurvey <- read.table(trainDataFrame, sep=",", header=TRUE, quote = '"')
  testData <- read.table(testDataFrame, sep=",", header=TRUE, quote = '"')


	numFolds <- 3

  model <- models[["glmer_binary"]]
  features <- list(
    LOG_IMDB_FEATURE,
    LOG_IMDB_NOSTEM_FEATURE,
    RATING_SIMILARITY_FEATURE,
    AVG_RATING_FEATURE,
    TAG_EXISTS_FEATURE,
    TAG_PROB,
    LSI_TAGS_75_FEATURE,
    LSI_IMDB_175_FEATURE
  )

  y <- dataSurvey$targets
  #y <- y$targets
  print("predictTagRelevance: Starting fitModel ...")
	#fit <- fitModel(dataSurvey, y, model, features)
  fit <- model$fitFunction(dataSurvey,y, getFeatureNames(features), model$transformY, model$excludeVals)
  print("predictTagRelevance: Model is fitted")

	testPredictions <-  model$predictFunction(fit, testData, getFeatureNames(features), model$untransformY)
	output = testData[c('tag', 'movieId')]
	output$score = scale01(testPredictions)
  write.table(output, file=savePredictionsPath, sep=",", col.names = FALSE, row.names = FALSE, fileEncoding = "utf-8")
  print("Save predictions into")
  print(savePredictionsPath)
}



predictTagRelevance <- function(trainDataFile, predictDataFile, predictionsFile, testMode=FALSE, evalMode=FALSE, samplingRatio=.05, smallData=FALSE) {
	library(lme4)
	source("models.R")
	source("dataset.R")
	source("tagfit.R")
	source("utils.R")
	source("crossval.R")
	source("step_regression.R")

	modelName <- "glmer_binary"

	numFolds <- 3

  if (testMode) {
    print("WARNING: TEST MODE, using pilot data")
    surveyDataset <- getDataset("pilot/features_and_responses_firstround_optimal.txt")
    fullDataset <- getDataset("pilot/features_all_optimal.txt")
  } else {
    surveyDataset <- read.delim(file=trainDataFile,head=TRUE)
    fullDataset <- read.delim(file=predictDataFile,head=TRUE)
  }

  ## compute tagfit feature
  tagFitFeatures <- list(
    RATING_SIMILARITY_FEATURE,
    LSI_IMDB_25_FEATURE,
    AVG_RATING_FEATURE)
  tagFitFunction <- fit_glmer
  tagPredictFunction <- predict_glmer
  dataTag <- buildDatasetWithFeatures(fullDataset, tagFitFeatures, includeUser=FALSE)
  dataSurveyTagFit <- buildDatasetWithFeatures(surveyDataset, tagFitFeatures, includeUser=FALSE)
  dataSurveyTagFit <-	scaleDataset(dataSurveyTagFit, dataTag, tagFitFeatures)
  dataTag <- scaleDataset(dataTag, dataTag, tagFitFeatures)
  print(paste('before tag fit', date()))
  if (smallData) {
    sampleMinNeg <- 1
  } else {
    sampleMinNeg <- 0
  }

  tagFit <- getTagFit(dataTag, fullDataset$tag_exists, fullDataset$tag_count, tagFitFunction, tagPredictFunction,
    getFeatureNames(tagFitFeatures), samplingRatio=samplingRatio, sampleMinNeg=sampleMinNeg)
  print(paste('completed tag fit', date()))

  model <- models[[modelName]]
  features <- list(
    LOG_IMDB_FEATURE,
    LOG_IMDB_NOSTEM_FEATURE,
    RATING_SIMILARITY_FEATURE,
    AVG_RATING_FEATURE,
    TAG_EXISTS_FEATURE,
    TAG_PROB,
    LSI_TAGS_75_FEATURE,
    LSI_IMDB_175_FEATURE
  )
  dataSurveyUnscaled <- buildDatasetWithFeatures(surveyDataset, features, includeUser=FALSE)
  dataSurveyUnscaled$tag_prob <- tagFit$predict(dataSurveyTagFit)
  dataSurvey <- scaleDataset(dataSurveyUnscaled, dataSurveyUnscaled, features)
  #dataSurvey <- read.table(file="/home/ms314/projects/movielens-tagnav/r/out/dataSurvey.csv", header=TRUE, sep=",")

  #trainDataPath="/home/ms314/projects/libdat/temp/train0.csv"
  #testDataPath = "/home/ms314/projects/libdat/temp/test0.csv"
  #savePredictionsPath = "/home/ms314/projects/libdat/temp/predictions_r0.txt"
  #print(trainDataPath)
  #print(testDataPath)
  #print(savePredictionsPath)
  #dataSurvey <- read.table(file=trainDataPath, header=TRUE, sep=",", quote = '"')
  # dataSurvey <- dataSurvey[sample(nrow(dataSurvey), 300), ]

  testData <- read.table(file=testDataPath, header=TRUE, sep=",", quote = '"')

  #dataSurvey <- subset(dataSurvey, select = -c(movieId))
	# Sanity check
	print('SANITY CHECK')
	fit <- model$fitFunction(dataSurvey,surveyDataset$survey_response, getFeatureNames(features), model$transformY,model$excludeVals)
	testPredictions <- model$predictFunction(fit, dataSurvey, getFeatureNames(features),model$untransformY)
	print(paste('fit model', date()))
	print(paste("mae using same fit function:", round(meanabserror(testPredictions, surveyDataset$survey_response),3)))

	y <- surveyDataset$survey_response
	if (evalMode) {
		print('WARNING: EVAL MODE')
		rm(dataTag)
		fits <- list()
		predictions <- numeric(nrow(dataSurvey))
		folds <- get_groups(numFolds, nrow(dataSurvey))
		foldIndex <- 0
		for (fold in folds) {
			foldIndex <- foldIndex + 1
			print(paste('****PROCESSING FOLD:',foldIndex))
			xFit <- dataSurvey[-fold,]
			yFit <- y[-fold]
			xEval <- dataSurvey[fold,]
			yEval <- y[fold]
			fit <- fitModel(xFit, yFit, model, features)
			fits[[foldIndex]]<-fit
			foldPredictions <- model$predictFunction(fit$optimalFit, xEval,getFeatureNames(fit$optimalFeatures), model$untransformY)
			print(paste('Features:',paste(getFeatureNames(fit$optimalFeatures),collapse=", ")))
			print(paste('MAE:', round(meanabserror(foldPredictions,yEval ),4)))
			predictions[fold] <- foldPredictions
		}
		summary <- paste('Overall MAE:',round(meanabserror(y, predictions),4), 'Overall MAE w/rounding:',round(meanabserror(y, round(predictions)),4))
		print(summary)
	} else {
    print("predictTagRelevance: Starting fitModel ...")
		fit <- fitModel(dataSurvey, y, model, features)
    print("predictTagRelevance: Model is fitted")
		surveyPredictions <- model$predictFunction(fit$optimalFit, dataSurvey, getFeatureNames(fit$optimalFeatures), model$untransformY)
		print(paste('Overall MAE:',round(meanabserror(y, surveyPredictions), 4), 'Overall MAE w/rounding:',round(meanabserror(y, round(surveyPredictions)), 4)))
		print("Let's scale to 0-1 interval")
		print(round(meanabserror(scale01(y), scale01(surveyPredictions)), 4))

		testPredictions <-  model$predictFunction(fit$optimalFit, testData, getFeatureNames(fit$optimalFeatures), model$untransformY)
    write.table(testPredictions, file=savePredictionsPath, sep=",", col.names = FALSE, row.names = FALSE, fileEncoding = "utf-8")
    print("Save predictions into")
    print(savePredictionsPath)
    print("Test data MAE:")
    print(meanabserror(testPredictions, testData$targets))

		# # Predict on full dataset
		# fullDataset$tag_prob <- tagFit$predict(dataTag)
		# print(paste('computed tagProb on full dataset', date()))
		# dataAll <- buildDatasetWithFeatures(fullDataset, fit$optimalFeatures, includeUser=FALSE)
		# rm(fullDataset)
		# dataAll <- scaleDataset(dataAll, dataSurveyUnscaled, fit$optimalFeatures)
		# predictions <- model$predictFunction(fit$optimalFit, dataAll, getFeatureNames(fit$optimalFeatures), model$untransformY)
		# results <- data.frame(movieId=dataAll$movieId, tag=dataAll$tag, tagRelevance=predictions)
		# if (testMode==TRUE) {
		# 	predictionsFile <- c(predictionsFile, "_TEST")
		# }
		# write.table(results, file=predictionsFile, row.names=FALSE, sep="\t")
		# print(paste('Wrote table', date()))
	}
}

scale01 <- function(vec){
  (vec - 1.0) / 4
}

fitModel <- function(x, y, model, possibleFeatures, holdOutSize=1/3, terminateOnOptimal=TRUE) {

	holdOutIndices <- sample(1:nrow(x), round(nrow(x)*holdOutSize))
	xHoldOut <- x[holdOutIndices,]
	yHoldOut <- y[holdOutIndices]
	xTrain <- x[-holdOutIndices,]
	yTrain <- y[-holdOutIndices]

	infinity <- 99999
	featuresUsed <- list()
	featuresRemaining <- possibleFeatures
	optimalFeatures <- list()
	optimalError <- infinity

  if (1 == 1){
    while (length(featuresRemaining)>0) {
      bestFeature <- NULL
      lowestError <- infinity
      for (feature in featuresRemaining) {
        featuresToTry <- featuresUsed
        featuresToTry[[length(featuresToTry)+1]] <- feature
        fit <- model$fitFunction(xTrain,yTrain, getFeatureNames(featuresToTry), model$transformY, model$excludeVals)
        errorVal <- evalModel(xHoldOut, yHoldOut, model, getFeatureNames(featuresToTry), fit, errorFunction=meanabserror)
        if (errorVal < lowestError) {
          lowestError <- errorVal
          bestFeature <- feature
          bestFit <- fit
        }
      }
      featuresUsed[[length(featuresUsed)+1]] <- bestFeature
      if (lowestError < optimalError) {
        optimalError <- lowestError
        optimalFeatures <- featuresUsed
      } else {
        if (terminateOnOptimal) {
          break
        }
      }
      temp <- featuresRemaining
      featuresRemaining <- list()
      for (feature in temp) {
        if (feature$featureName !=bestFeature$featureName) {
          featuresRemaining[[length(featuresRemaining)+1]] <- feature
        }
      }

      featuresRemaining <- removeFeaturesFromSameGrouping(featuresRemaining, featuresUsed)
      # #print(paste(round(lowestError,4), paste(getFeatureNames(featuresUsed), collapse=", "), date()))
    }
  }
	# start my
	#f_lst <- c("log_IMDB", "tag_prob", "lsi_tags_75", "tag_exists", "rating_similarity", "lsi_imdb_175", "log_IMDB_nostem", "avg_rating")
	#optF <- character(length(f_lst))
	#for (i in 1:length(f_lst)) {optF[i] <- f_lst[i]}
	#fit <- model$fitFunction(x,y, optF, model$transformY, model$excludeVals)
	# end my
  print("fitModel: cycle is finished")
  print("fitModel: optimal features are")
  print(getFeatureNames(optimalFeatures))
  print("fitModel: Start make final fit")
  #browser()
  fit <- model$fitFunction(x,y, getFeatureNames(optimalFeatures), model$transformY, model$excludeVals)
  print("fitModel: End make final fit")
  list(optimalFeatures=optimalFeatures, optimalFit=fit)
}

evalModel <- function(xEval, yEval, model, featuresUsed, fit, errorFunction=meanabserror) {
	yPredicted <- model$predictFunction(fit, xEval, featuresUsed, model$untransformY)
	errorFunction(yPredicted, yEval)
}


get_groups <- function (ngroup, n) {
    if(ngroup==n) {groups <- 1:n; leave.out <- 1}
    if(ngroup<n){
        leave.out <- trunc(n/ngroup);
        o <- sample(1:n)
        groups <- vector("list",ngroup)
        for(j in 1:(ngroup-1)){
            jj <- (1+(j-1)*leave.out)
            groups[[j]] <- (o[jj:(jj+leave.out-1)])
        }
        groups[[ngroup]] <- o[(1+(ngroup-1)*leave.out):n]
    }
    groups
}

# survey features
surveyFilePath  <- "../../../data/processed/features_r.csv"
# features of items, for which we want to generate scores
featureFilePath   <- "../../../data/processed/features_for_score_generation.csv"
# output file path
outputFilePath   <- "../../../data/processed/glmer.csv"
generateScores(surveyFilePath, 
               featureFilePath, 
               outputFilePath)
