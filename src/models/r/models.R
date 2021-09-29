#library('e1071')
source('utils.R')

# SVM

fit_svm_nopool <- function(x, y, features, transformY=identity, excludeVals=c(),  weights=NULL, c=1, kernel="polynomial") {
# This requires that data be binarized somehow

	z <- transformY(exclude(excludeVals,y))
	binaryToClass = c("F", "T")
	z <- factor(binaryToClass[z + 1]) # 0->F, 1->T

	tags <- unique(x$tag)
	fit <- list()
	for (tag in tags) {
		hasTag <- x$tag==tag
		zTag <- z[hasTag]
		xTag <- x[hasTag,]
		if (!is.null(weights)) {
			numPositive = sum(hasTag & z=="T")
			numNegative = sum(hasTag & z=="F")
			positiveWeight = sum(weights[hasTag & z=="T"])
			negativeWeight = sum(weights[hasTag & z=="F"])
			posClassWeight = positiveWeight / numPositive
			negClassWeight = negativeWeight / numNegative
		} else {
			posClassWeight = 1
			negClassWeight = 1
		}
		#print(tag)
		fit[[tag]] <- svm(x=xTag[features], y=zTag, kernel=kernel, probability=TRUE, cost=c, class.weights=list("F"=negClassWeight,"T"=posClassWeight))
	}
	fit
}

predict_svm_nopool <- function(fit,x, features, untransformY=identity, decisionValues = FALSE, ...) {
	tags <- unique(x$tag)
	predicted <- numeric(nrow(x))
	naRows <- rowSums(is.na(x[features])) > 0
	predicted[naRows] <- NA
	for (tag in tags){
		hasTag <- x$tag==tag
#		predicted[hasTag & !naRows] <- untransformY(attr(predict(fit[[tag]], x[hasTag & !naRows,][features], probability=TRUE),"probabilities")[,2])
		tagPredicted <- predict(fit[[tag]], x[hasTag & !naRows,][features], decision.values=decisionValues)
		if (decisionValues) {
			output <- attr(tagPredicted, "decision.values")[,1] # Raw output of svm
		} else {
			output <- as.numeric(tagPredicted) - 1 # Convert T,F to 1,0
		}
		predicted[hasTag & !naRows] <- untransformY(output)
	}
	#browser()
	predicted
}

fit_svm_nopool_prob <- function(x, y, features, transformY=identity, excludeVals=c(), weights=NULL, c=1, kernel="radial") {
# This requires that data be binarized somehow
	svmFit <- fit_svm_nopool(x, y, features, transformY=identity, excludeVals=excludeVals, weights, c, kernel)
	x$svm_predictions <- predict_svm_nopool(svmFit, x, features, untransformY=identity, decisionValues=TRUE)
	logisticFit <- fit_glm_nopool(x,y=y, features=c("svm_predictions"), transformY=identity, weights=weights)
	list(svmFit=svmFit, logisticFit=logisticFit)
}

predict_svm_nopool_prob <- function(fit,x, features, untransformY=identity, ...) {
	x$svm_predictions <- predict_svm_nopool(fit$svmFit, x, features, untransformY=identity, decisionValues=TRUE)
	predict_glm_nopool(fit$logisticFit, x, c("svm_predictions"), untransformY=identity)
}

# Generalized linear models

fit_glmer <- function(x,y, features, transformY, excludeVals=c(), family=binomial(link="logit"), control=glmerControl(optCtrl=list(maxfun=500)), weights=NULL,...) {
  z <- transformY(exclude(excludeVals, y))
  # z[1] <- 0.9 # for testing in case if all values are the same
	form <- paste("z ~ ", paste(features, collapse=" + "))
	for (feature in features) {
		form <- paste(form, " + (", feature," - 1|tag)",sep="")
	}
	form <- paste(form, "+ (1|tag)")
	form <- formula(form)
	if (is.null(weights)) {
		if (is.null(control)) {
			fit <- glmer(form, data=x, family=family)
		} else {
			fit <- glmer(form, data=x, family=family, control=control)
		}
	} else {
		fit <- glmer(form, data=x, weights=weights, family=family, control=control)
	}
	predictedY <- predict_glmer(fit,x, features, untransform_identity)
	attr(fit, "secondaryFit") <- lm (y ~ predictedY)
#	attr(fit, "secondaryFit") <- rq (y ~ predictedY, tau = .5)
	fit
}
predict_glmer <- function(fit,x, features, untransformY, family=NULL, control=NULL, ...) {
	n <- nrow(x)
	fe <- fixef(fit)
	re <- ranef(fit)$tag
	predictions <- rep(NA,n)
	for (i in 1:n) {
		tag <- as.character(x[i,"tag"])
		randEff <- re[tag,"(Intercept)"]
		result <- (ifelse(is.na(randEff), 0, randEff) + fe["(Intercept)"])*1
		for (feature in features) {
			randEff <- re[tag,feature]
			result <- result + (ifelse(is.na(randEff), 0, randEff) + fe[feature])*x[i,feature]
		}
		predictions[i] <- logistic(result)
	}
	untransformY(predictions, fit)
}


fit_glm_nopool <- function(x,y, features, transformY, excludeVals=c(), family=quasibinomial(link="logit"), control = glm.control(maxit=1000) ,weights=NULL,...) {
	z <- transformY(exclude(excludeVals,y))
	tags <- unique(x$tag)
	fit <- list()
	for (tag in tags) {
		hasTag <- x$tag==tag
		if (sum(hasTag & !is.na(z)) > 1 ) {
			zTag <- z[hasTag]
			xTag <- x[hasTag,]
			form <- paste("zTag ~", paste(features, collapse=" + "))
			form <- formula(form)

			if (is.null(weights)) {
				if (is.null(control)) {
					fit[[tag]] <- glm(form, data=xTag, family=family)
				} else {
					fit[[tag]] <- glm(form, data=xTag, family=family,control=control)
				}
			} else {
				fit[[tag]] <- glm(form, data=xTag, weights=weights[hasTag], family=family,control=control)
			}
		}
	}
	fit
}

predict_glm_nopool <- function(fit,x, features, untransformY, family=NULL, control=NULL, ...) {
	tags <- unique(x$tag)
	predicted <- numeric(nrow(x))
	for (tag in tags){
		if (is.null(fit[[tag]])) {
			prediction <- .5
		} else {
			prediction <- predict(fit[[tag]], x[x$tag==tag,], type="response")
		}
		predicted[x$tag==tag] <- untransformY(prediction)
	}
	predicted
}

fit_glm <- function(x,y, features, transformY, excludeVals=c(), family=quasibinomial(link="logit"), control=NULL) {
	z <- transformY(exclude(excludeVals,y))
	form <- paste("z ~", paste(features, collapse=" + "))
	fit <- glm(formula(form), family=family, data=x)
	predictedY <- predict_glm(fit,x, features, untransform_identity)
	attr(fit, "secondaryFit") <- lm (y ~ predictedY)
	fit
}
predict_glm <- function(fit, x, features, untransformY, family=NULL, control=NULL, ...) {
	untransformY(predict(fit, x, type="response"), fit)
}

# Linear models

fit_lmer <- function(x,y, features, transformY, excludeVals=c(), weights=NULL,...) {
	z <- transformY(exclude(excludeVals,y))
	form <- paste("z ~", paste(features, collapse=" + "))
	for (feature in features) {
		form <- paste(form, " + (", feature," - 1|tag)",sep="")
	}
	form <- paste(form, "+ (1|tag)")
	if (is.null(weights)) {
		lmer(formula(form), data=x)
	} else {
		lmer(formula(form), data=x, weights=weights)
	}
}
fit_lmer_dependent <- function(x,y, features, transformY, excludeVals=c(), weights=NULL,...) {
	z <- transformY(exclude(excludeVals,y))
	form <- paste("z ~", paste(features, collapse=" + "))
	form <- paste(form, " + (", paste(features, collapse=" + "), " + 1|tag)")
	#print(form)
	if (is.null(weights)) {
		lmer(formula(form), data=x)
	} else {
		lmer(formula(form), data=x, weights=weights)
	}
}
predict_lmer <- function(fit,x, features, untransformY, ...) {
	n <- nrow(x)
	fe <- fixef(fit)
	re <- ranef(fit)$tag
	predictions <- rep(NA,n)
	for (i in 1:n) {
		tag <- as.character(x[i,"tag"])
		randEff <- re[tag,"(Intercept)"]
		result <- (ifelse(is.na(randEff), 0, randEff) + fe["(Intercept)"])*1
		for (feature in features) {
			randEff <- re[tag,feature]
			result <- result + (ifelse(is.na(randEff), 0, randEff) + fe[feature])*x[i,feature]
		}


		predictions[i] <- result
	}
	untransformY(predictions)
}

fit_lm_nopool <- function(x,y, features, transformY, excludeVals=c(), weights=NULL,...) {
	z <- transformY(exclude(excludeVals,y))
	tags <- unique(x$tag)
	fit <- list()
	for (tag in tags) {
		hasTag <- x$tag==tag
		if (sum(hasTag & !is.na(z)) > 1 ) {
			zTag <- z[hasTag]
			xTag <- x[hasTag,]
			form <- paste("zTag ~", paste(features, collapse=" + "))
			form <- formula(form)
			if (is.null(weights)) {
				fit[[tag]] <- lm(form, data=xTag)
			} else {
				fit[[tag]] <- lm(form, data=xTag, weights=weights[hasTag])
			}
		}
	}
	fit
}

predict_lm_nopool <- function(fit,x, features, untransformY, ...) {
	tags <- unique(x$tag)
	predicted <- numeric(nrow(x))
	for (tag in tags){
		if (is.null(fit[[tag]])) {
# This assumes that is on 0,1 scale!
			prediction <- .5
		} else {
			prediction <- predict(fit[[tag]], x[x$tag==tag,], type="response")
		}
		predicted[x$tag==tag] <- untransformY(prediction)
	}
	predicted
}
fit_lm <- function(x,y, features, transformY, excludeVals=c()) {
	z <- transformY(exclude(excludeVals,y))
	form <- paste("z ~", paste(features, collapse=" + "))
	lm(formula(form), data=x)
}
predict_lm <- function(fit, x, features, untransformY, ...) {
	untransformY(predict(fit, x, type="response"))
}

# Method to convert certain y values to NA

exclude <- function(excludeVals, y) {
	ifelse(y %in% excludeVals, NA, y)
}

# Transformations of response variable to aid in model fitting and prediction

transform_binary <- function (y, middleVal = 1, ...) {
	z <- c()
	for (y_i in y) {
		if (is.na(y_i) | is.null(y_i)) {
			z_i <- y_i
		} else {
			if (y_i==4 | y_i==5) {
				z_i <- 1
			}
			if (y_i==1 | y_i==2) {
				z_i <- 0
			}
			if (y_i==3) {
				z_i <- middleVal
			}
		}
		z <- c(z, z_i)
	}
	z
}

transform_binary_middle_0 <- function (y) {
	transform_binary(y, middleVal=0)
}


untransform_binary <- function (y, ...) {
	(4 * y) + 1
}

transform_continuous_01 <- function(y, ...) {
	(y - 1)/4
}
untransform_continuous_01 <- function(y, ...) {
	(4 * y) + 1
}
transform_identity <- function(y, ...) {
	y
}
untransform_identity <- function(y, ...) {
	y
}
untransform_secondary_fit <- function(y, fit, ...) {
	secondaryFit <- attr(fit, "secondaryFit")
	predict(secondaryFit, data.frame(predictedY=y), type="response")
}
clip <- function(y, minVal, maxVal) {
	newY <- c()
	for (yVal in y) {
		newVal <- yVal
		if (newVal < minVal) {
			newVal <- minVal
		}
		if (newVal > maxVal) {
			newVal <- maxVal
		}
		newY <- c(newY, newVal)
	}
	newY
}

# Assorted methods

get_mer_coefs <- function(fit) {
	fe <- fixef(fit)
	re <- ranef(fit)$tag
	coefs <- re
	features <- colnames(re) #Includes constant term
	for (i in 1:nrow(re)) {
		for (feature in features) {
			coefs[i,feature] <- re[i,feature] + fe[feature]
		}
	}
	coefs
}

get_mer_desc <- function(x, y, features, model, tags) {
	fit <- model$fitFunction(x,y, features, model$transformY)
	modelFitDesc <- list()
	fe <- fixef(fit)
	re <- ranef(fit)$tag
	terms <- c()
	for (feature in features) {
		term <- round(fe[feature],3)
		terms <- c(terms, term)
	}
	term <- round(fe["(Intercept)"])
	terms <- c(terms, term)
	modelFitDesc[['fixed_effects']] <- paste(terms, collapse="\t")

	for (tag in tags) {
		terms <- c()
		for (feature in features) {
#			term <- paste(round(re[tag,feature] + fe[feature],3), feature, sep="*")
			term <- round(re[tag,feature] + fe[feature],3)
			terms <- c(terms, term)
		}
		term <- round(re[tag,"(Intercept)"] + fe["(Intercept)"],3)
		terms <- c(terms, term)
		modelFitDesc[[tag]] <- paste(terms, collapse="\t")
	}
	modelFitDesc
}

fitAndPredict <- function(x, y, features, model) {
	fit <- model$fitFunction(x,y, features, model$transformY, model$excludeVals)
	model$predictFunction(fit, x, features, model$untransformY)
}

# List of models

models = list()
models[["glmer"]] <- list(fitFunction=fit_glmer, predictFunction=predict_glmer, transformY=transform_continuous_01, untransformY=untransform_continuous_01, excludeVals=c())
models[["glmer_binary"]] <- list(fitFunction=fit_glmer, predictFunction=predict_glmer, transformY=transform_binary, untransformY=untransform_binary, excludeVals=c(3))
models[["glmer_binary_fit"]] <- list(fitFunction=fit_glmer, predictFunction=predict_glmer, transformY=transform_binary, untransformY=untransform_secondary_fit, excludeVals=c(3))
models[["glm_nopool"]] <- list(fitFunction=fit_glm_nopool, predictFunction=predict_glm_nopool, transformY=transform_continuous_01, untransformY=untransform_continuous_01, excludeVals=c())
models[["glm_nopool_binary"]] <- list(fitFunction=fit_glm_nopool, predictFunction=predict_glm_nopool, transformY=transform_binary, untransformY=untransform_binary, excludeVals=c(3))
models[["lmer"]] <- list(fitFunction=fit_lmer, predictFunction=predict_lmer, transformY=identity, untransformY=identity, excludeVals=c())
models[["lmer_binary"]] <- list(fitFunction=fit_lmer, predictFunction=predict_lmer, transformY=transform_binary, untransformY=untransform_binary, excludeVals=c(3))
models[["lm_nopool"]] <- list(fitFunction=fit_lm_nopool, predictFunction=predict_lm_nopool, transformY=identity, untransformY=identity, excludeVals=c())
#If use the above model, then must fix lm_nopool_predict (use 3 instead of .5)
models[["lm_nopool_binary"]] <- list(fitFunction=fit_lm_nopool, predictFunction=predict_lm_nopool, transformY=transform_binary, untransformY=untransform_binary, excludeVals=c(3))
models[["lm_complete"]] <- list(fitFunction=fit_lm, predictFunction=predict_lm, transformY=identity, untransformY=identity, excludeVals=c())
models[["lm_complete_binary"]] <- list(fitFunction=fit_lm, predictFunction=predict_lm, transformY=transform_binary, untransformY=untransform_binary, excludeVals=c(3))
models[["glm_complete"]] <- list(fitFunction=fit_glm, predictFunction=predict_glm, transformY=transform_continuous_01, untransformY=untransform_continuous_01, excludeVals=c())
models[["glm_complete_binary"]] <- list(fitFunction=fit_glm, predictFunction=predict_glm, transformY=transform_binary, untransformY=untransform_binary, excludeVals=c(3))
models[["glm_complete_binary_fit"]] <- list(fitFunction=fit_glm, predictFunction=predict_glm, transformY=transform_binary, untransformY=untransform_secondary_fit, excludeVals=c(3))
