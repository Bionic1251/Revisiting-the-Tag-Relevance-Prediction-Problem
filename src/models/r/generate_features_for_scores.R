
generateFeaturesForScores <- function(featureFilePath,
                                itemIdsFilePath,
                                surveyFilePath,
                                outputTrainingFilePath,
                                outputFeatureFilePath,
                                samplingRatio=.05) {
	# part of predictTagRelevance  which calculates features
	library(lme4)
	source("models.R")
	source("dataset.R")
	source("tagfit.R")
	source("utils.R")
	source("crossval.R")
	source("step_regression.R")

	fullDataset <- read.delim(file=featureFilePath,head=TRUE)
	# removing movie ids outside specified ones
	itemIds = read.delim(file=itemIdsFilePath,head=TRUE)
	fullDataset = fullDataset[fullDataset$movieId %in% itemIds$movie_id ,]
	# removing tags outside the survey
	dataSurvey <- read.table(surveyFilePath, sep=",", header=TRUE, quote = '"')
	fullDataset = fullDataset[fullDataset$tag %in% dataSurvey$tag ,]
  
  # compute tagfit feature
  tagFitFeatures <- list(
    RATING_SIMILARITY_FEATURE,
    LSI_IMDB_25_FEATURE,
    AVG_RATING_FEATURE)
  tagFitFunction <- fit_glmer
  tagPredictFunction <- predict_glmer
  dataTag <- buildDatasetWithFeatures(fullDataset, tagFitFeatures, includeUser=FALSE)
  dataTag <- scaleDataset(dataTag, dataTag, tagFitFeatures)
  print(paste('before tag fit', date()))
  sampleMinNeg <- 0
  # browser()
  tagFit <- getTagFit(dataTag, fullDataset$tag_exists, fullDataset$tag_count, tagFitFunction, tagPredictFunction,
    getFeatureNames(tagFitFeatures), samplingRatio=samplingRatio, sampleMinNeg=sampleMinNeg)
  print(paste('completed tag fit', date()))
  
  fullDataset$tag_prob <- tagFit$predict(dataTag)
  print(paste('Tag prob feature generated', date()))
	
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

	dataUnscaled <- buildDatasetWithFeatures(fullDataset, features, includeUser=FALSE)
	trainingData = merge(x = dataSurvey[,c("tag", "movieId", "targets")], y = dataUnscaled, by = c("movieId", "tag"), all.x = TRUE)
	print(paste('dataset is constructed', date()))
	data <- scaleDataset(dataUnscaled, trainingData, features)
	print(paste('dataset is scaled', date()))
	trainingData = merge(x = dataSurvey[,c("tag", "movieId", "targets")], y = data, by = c("movieId", "tag"), all.x = TRUE)
	write.table(trainingData, file=outputTrainingFilePath, sep=",", col.names = TRUE, row.names = FALSE, fileEncoding = "utf-8", quote=FALSE)
	write.table(data, file=outputFeatureFilePath, sep=",", col.names = TRUE, row.names = FALSE, fileEncoding = "utf-8", quote=FALSE)
}


# Version with additional functionality
predictTagRelevance <- function(trainDataFile, predictDataFile, predictionsFile, dataSurveyInterimFile, recalculateDataSurvey=TRUE, testMode=FALSE, evalMode=FALSE, samplingRatio=.05, smallData=FALSE) {
	library(lme4)
	source("models.R")
	source("dataset.R")
	source("tagfit.R")
	source("utils.R")
	source("crossval.R")
	source("step_regression.R")

	modelName <- "glmer_binary"

	numFolds <- 5 # Only matters when evalMode set to true

	if (testMode) {
		print("WARNING: TEST MODE, using pilot data")
		surveyDataset <- getDataset("pilot/features_and_responses_firstround_optimal.txt")
		fullDataset <- getDataset("pilot/features_all_optimal.txt")
	} else {
		surveyDataset <- read.delim(file=trainDataFile,head=TRUE)
		fullDataset <- read.delim(file=predictDataFile,head=TRUE)
	}
  
  if (recalculateDataSurvey){
    # compute tagfit feature
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
  }

	
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
  
  if (recalculateDataSurvey){
    # Sanity check
    print('SANITY CHECK')
    fit <- model$fitFunction(dataSurvey,surveyDataset$survey_response, getFeatureNames(features), model$transformY,model$excludeVals)
    testPredictions <- model$predictFunction(fit, dataSurvey, getFeatureNames(features),model$untransformY)
    print(paste('fit model', date()))
    print(paste("mae using same fit function:", round(meanabserror(testPredictions, surveyDataset$survey_response),3)))
  }

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

		# Fit model
		fit <- fitModel(dataSurvey, y, model, features)

    # Dump to file to be used by models
    dataSurveyWithTarget <- dataSurvey
    dataSurveyWithTarget$targets <- y
    write.table(dataSurveyWithTarget, file=dataSurveyInterimFile, sep=",", col.names = TRUE, row.names = FALSE, fileEncoding = "utf-8", quote=FALSE)
    print(paste("Save dataSurvey into", dataSurveyInterimFile))

    dataSurvey <- read.table(file=dataSurveyInterimFile, header=TRUE, sep=",", quote = '"')


		# As a check, evaluate on the survey data
		surveyPredictions <- model$predictFunction(fit$optimalFit, dataSurvey, getFeatureNames(fit$optimalFeatures), model$untransformY)
		print(paste('Overall MAE:',round(meanabserror(y, surveyPredictions),4), 'Overall MAE w/rounding:',round(meanabserror(y, round(surveyPredictions)),4)))

		# Predict on full dataset
		fullDataset$tag_prob <- tagFit$predict(dataTag)
		print(paste('computed tagProb on full dataset', date()))
		dataAll <- buildDatasetWithFeatures(fullDataset, fit$optimalFeatures, includeUser=FALSE)
		rm(fullDataset)
		dataAll <- scaleDataset(dataAll, dataSurveyUnscaled, fit$optimalFeatures)
		predictions <- model$predictFunction(fit$optimalFit, dataAll, getFeatureNames(fit$optimalFeatures), model$untransformY)
		results <- data.frame(movieId=dataAll$movieId, tag=dataAll$tag, tagRelevance=predictions)
		if (testMode==TRUE) {
			predictionsFile <- c(predictionsFile, "_TEST")
		}
		write.table(results, file=predictionsFile, row.names=FALSE, sep="\t")
		print(paste('Wrote table', date()))
	}
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
		#print(paste(round(lowestError,4), paste(getFeatureNames(featuresUsed), collapse=", "), date()))
	}
	fit <- model$fitFunction(x,y, getFeatureNames(optimalFeatures), model$transformY, model$excludeVals)
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

# item features
featureFilePath  <- "../../../data/processed/predict_relevance.txt"
# item ids, for which we want to generate scores
itemIdsFilePath   <- "../../../data/processed/movie_ids.csv"
# output file path for features
outputFeatureFilePath   <- "../../../data/processed/features_for_score_generation.csv"
# output file path for training data
outputTrainingFilePath   <- "../../../data/processed/training_data_for_score_generation.csv"
# features generated based on survey answers. We use it to filter tags
surveyFilePath  <- "../../../data/processed/features_r.csv"
generateFeaturesForScores(featureFilePath, itemIdsFilePath, surveyFilePath, outputTrainingFilePath, outputFeatureFilePath)

