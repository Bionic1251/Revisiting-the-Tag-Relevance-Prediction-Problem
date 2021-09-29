source("utils.R")
stepRegression <- function(x, y, model, features, errorFunction=meanabserror, separateTagsInCrossVal=FALSE, terminateOnOptimal=FALSE,
 		verbose=TRUE, ngroup=10, allowGroupingOverlap=FALSE, featuresAlwaysInclude=list()) {
	source("crossval.R")
	infinity <- 99999
	featuresUsed <- featuresAlwaysInclude
	featuresRemaining <- features
	optimalFeatures <- list()
	optimalError <- infinity
	while (length(featuresRemaining)>0) {
		bestFeature <- NULL
		lowestError <- infinity
		for (feature in featuresRemaining) {
			featuresToTry <- featuresUsed
			featuresToTry[[length(featuresToTry)+1]] <- feature
			if (separateTagsInCrossVal) {
				predicted <- cross_val_group(x, y, model, groupingValues=x$tag, ngroup=ngroup, features=getFeatureNames(featuresToTry))$cv.fit
			} else {
				predicted <- cross_val(x, y, model, ngroup=ngroup, features=getFeatureNames(featuresToTry))$cv.fit
			}
			errorVal <- errorFunction(y, predicted)
			if (errorVal < lowestError) {
				lowestError <- errorVal
				bestFeature <- feature
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
		if (!allowGroupingOverlap) {
			featuresRemaining <- removeFeaturesFromSameGrouping(featuresRemaining, featuresUsed)
		}
		if (verbose) {
			print(paste(round(lowestError,4), paste(getFeatureNames(featuresUsed), collapse=", "), date()))
		}
	}
	list(optimalFeatures=getFeatureNames(optimalFeatures), optimalError=optimalError)		
}
