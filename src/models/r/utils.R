logistic <- function(t) {
	1/(1+exp(-t))
}

meanabserror <- function(actual, predicted){
	mean(abs(actual - predicted))
}

rootmeansquarederror <- function(actual, predicted){
	sqrt(mean((actual - predicted)^2))
}

loglikelihood <- function(actual, predicted) {
# Assumes that predicted contains the probability of positive class (1)
	probVector <- numeric(length(actual))
	probVector[actual==1] <- predicted[actual==1]
	probVector[actual==0] <- 1 - predicted[actual==0]
	sum(log(probVector))
}

getAccuracy <- function(actual, predicted) {
# actual and predicted must be binary vectors
	numCorrect <- sum(actual==predicted)
	n <- length(actual)
	numCorrect / n
}

getPrecision <- function(actual, predicted) {
# actual and predicted must be binary vectors
	numPositiveCorrect <- sum(predicted==1 & actual==1)
	numPositivePredicted <- sum(predicted==1)
	numPositiveCorrect / numPositivePredicted 
}

getRecall <- function(actual, predicted) {
# actual and predicted must be binary vectors
	numPositiveCorrect <- sum(predicted==1 & actual==1)
	numTruePositive <- sum(actual==1)
	numPositiveCorrect / numTruePositive
}

fMeasure <- function(precision, recall) {
	2 * precision * recall / (precision + recall)
}

getBetaParams <- function(popMean, popVariance, minVal=0, maxVal=1) {
	popMean = (popMean - minVal)/(maxVal-minVal)
	popVariance = popVariance/((maxVal - minVal)^2)
	print(paste("mean", popMean, "var", popVariance))
	alpha <- popMean*(popMean*(1-popMean)/popVariance - 1)
	beta <- (1-popMean)*alpha
	list(alpha=alpha, beta=beta)
}

