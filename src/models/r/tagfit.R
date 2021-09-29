NUM_ITER <- 1

getTagFit <- function(dataTag, tagExists, posWeight, fitFunction, predictFunction, features, c=NULL, kernel=NULL, samplingRatio = 1, sampleMinNeg=0) {

	tags <- unique(dataTag$tag)

	if (samplingRatio < 1) {
#		print(paste('WARNING: sampling ratio =', samplingRatio))
		positiveIndices <- which(tagExists == 1)
		negativeIndicesKeep <- c()
		for (tag in tags) {
			negativeIndicesTag <- which(tagExists==0 & dataTag$tag==tag)
			nNegative <- length(negativeIndicesTag)
			nInSample <- round(samplingRatio * nNegative)

			# if sampleMinNeg is set, then sample at least sampleMinNeg when possible
			if (nInSample < sampleMinNeg) {
			  if (nNegative < sampleMinNeg) {
			    nInSample <- nNegative
			  } else {
			    nInSample <- sampleMinNeg
			  }
			}
			negativeIndicesKeep <- c(negativeIndicesKeep, sample(negativeIndicesTag, nInSample))
		}
		indicesToKeep <- c(positiveIndices, negativeIndicesKeep)
		dataTag <- dataTag[indicesToKeep,]
		tagExists <- tagExists[indicesToKeep]
		posWeight <- posWeight[indicesToKeep]
	}
	y <- tagExists
	
# Initialize weights for positive examples.  These won't change.
	weights <- posWeight

# Initialize weights for unlabeled(negative) examples.  These will change.

	for (tag in tags) {
		numPositive <- sum(weights[dataTag$tag==tag & y==1])
		unlabeled <- dataTag$tag==tag & y==0
		numUnlabeled <- sum(unlabeled)
		weights[unlabeled] <- numPositive / numUnlabeled
	}

# Employ expectation maximization algorithm
# ACTUALLY, just eliminate likely false-positives

	if (NUM_ITER > 1 ) {
		print(paste('WARNING:', NUM_ITER, 'iterations'))
	}
	for (i in 1:NUM_ITER) {

		#print(paste('Completed iteration', i, date()))
		# Fit model
		if (!is.null(c) | !is.null(kernel) ) { 
			fit <- fitFunction(dataTag,y, features=features, transformY=transform_identity, weights=weights, c=c, kernel=kernel)
		} else {
			fit <- fitFunction(dataTag,y, features=features, transformY=transform_identity, weights=weights)
		}
		# Estimate p (probability of belonging to positive class) using model fit
		p <- predictFunction(fit, dataTag, features, untransformY=untransform_identity)	
		# Generate probabilistically-weighted negative examples based on p (probability of belonging to positive class)	

		for (tag in tags) {
			numPositive <- sum(weights[dataTag$tag==tag & y==1])
			unlabeled <- dataTag$tag==tag & y==0
			weights[unlabeled] <- ifelse(p[unlabeled]>.5, 0, weights[unlabeled])
			##if (i < NUM_ITER -1) {
			#	# Version that does NOT equally weight positive and negative
			#	weights[unlabeled] <- (1 - p[unlabeled]) * (numPositive / sum(p[dataTag$tag==tag & !is.na(p)]))
			##} else {
			##	weights[unlabeled] <- (1 - p[unlabeled]) * (numPositive / sum(1 - p[dataTag$tag==tag & !is.na(p)]))
			##}
		}
		keepIndices <- !is.na(weights) & weights > 0
		dataTag <-dataTag[keepIndices,]
		y <-y[keepIndices]
		weights <- weights[keepIndices]
		
		for (tag in tags) {
			numPositive <- sum(weights[dataTag$tag==tag & y==1])
			unlabeled <- dataTag$tag==tag & y==0
			numUnlabeled <- sum(unlabeled)
			weights[unlabeled] <- numPositive / numUnlabeled
		}
#		print(paste("iter", i, date()))
	}	

	# Return predict function

	predict <- function(dataPredict){predictFunction(fit, dataPredict, features, untransformY=untransform_identity)} 
	list(predict=predict, fit=fit)
	
}