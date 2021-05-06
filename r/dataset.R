IMDB_FEATURE <- list(colName='term_freq_IMDB', transform=identity, featureName='IMDB') 
LOG_IMDB_FEATURE <- list(colName='term_freq_IMDB_log', transform=identity, featureName='log_IMDB') 
LOG_IMDB_NOSTEM_FEATURE <- list(colName='term_freq_IMDB_log_nostem', transform=identity, featureName='log_IMDB_nostem')
LSI_IMDB_FEATURE <- list(colName='lsi_imdb', transform=identity, featureName='lsi_imdb') 
TAG_EXISTS_FEATURE <- list(colName='tag_exists', transform=identity, featureName='tag_exists') 
LSI_TAGS_FEATURE <- list(colName='lsi_tags', transform=identity, featureName='lsi_tags') 
RATING_SIMILARITY_FEATURE <- list(colName='tag_movie_rating_similarity', transform=identity, featureName='rating_similarity') 
AVG_RATING_FEATURE <- list(colName='avg_movie_rating', transform=identity, featureName='avg_rating') 
AVG_RATING_FEATURE_SQUARED <- list(colName='avg_movie_rating', transform=function(x){x^2}, featureName='avg_rating_squared') 
ADJ_USER_RATING_FEATURE <- list(colName='movie_rating', transform=identity, featureName='adjusted_user_rating')
TAG_PROB <- list(colName='tag_prob', transform=identity, featureName='tag_prob')
SURVEY_RESPONSE <- list(colName='survey_response', transform=identity, featureName='y')
TAG_LOGISTIC <- list(colName='tag_logistic', transform=identity, featureName='tag_logistic')
LSI_IMDB_25_FEATURE <- list(colName='lsi_imdb_25', transform=identity, featureName='lsi_imdb_25', group='lsi_imdb') 
LSI_IMDB_50_FEATURE <- list(colName='lsi_imdb_50', transform=identity, featureName='lsi_imdb_50', group='lsi_imdb') 
LSI_IMDB_75_FEATURE <- list(colName='lsi_imdb_75', transform=identity, featureName='lsi_imdb_75', group='lsi_imdb') 
LSI_IMDB_100_FEATURE <- list(colName='lsi_imdb_100', transform=identity, featureName='lsi_imdb_100', group='lsi_imdb') 
LSI_IMDB_125_FEATURE <- list(colName='lsi_imdb_125', transform=identity, featureName='lsi_imdb_125', group='lsi_imdb') 
LSI_IMDB_150_FEATURE <- list(colName='lsi_imdb_150', transform=identity, featureName='lsi_imdb_150', group='lsi_imdb') 
LSI_IMDB_175_FEATURE <- list(colName='lsi_imdb_175', transform=identity, featureName='lsi_imdb_175', group='lsi_imdb') 
LSI_IMDB_200_FEATURE <- list(colName='lsi_imdb_200', transform=identity, featureName='lsi_imdb_200', group='lsi_imdb') 
LSI_TAGS_25_FEATURE <- list(colName='lsi_tags_25', transform=identity, featureName='lsi_tags_25', group='lsi_tags')
LSI_TAGS_50_FEATURE <- list(colName='lsi_tags_50', transform=identity, featureName='lsi_tags_50', group='lsi_tags')
LSI_TAGS_75_FEATURE <- list(colName='lsi_tags_75', transform=identity, featureName='lsi_tags_75', group='lsi_tags')
LSI_TAGS_100_FEATURE <- list(colName='lsi_tags_100', transform=identity, featureName='lsi_tags_100', group='lsi_tags')
LSI_TAGS_125_FEATURE <- list(colName='lsi_tags_100', transform=identity, featureName='lsi_tags_125', group='lsi_tags')
LSI_TAGS_150_FEATURE <- list(colName='lsi_tags_150', transform=identity, featureName='lsi_tags_150', group='lsi_tags')
LSI_TAGS_175_FEATURE <- list(colName='lsi_tags_175', transform=identity, featureName='lsi_tags_175', group='lsi_tags')
LSI_TAGS_200_FEATURE <- list(colName='lsi_tags_200', transform=identity, featureName='lsi_tags_200', group='lsi_tags')

getDataset <- function(fileName) {
	dir <- paste(Sys.getenv("TAGNAV_FILES_DIR"), 'temp/r/data/', sep="")
	path <- paste(dir, fileName, sep="")
	read.delim(file=path,head=TRUE)
}

# Requires tag, movieId in dataset
buildDatasetWithFeatures <- function(dataset, features, includeUser=FALSE) {
	newDataset <- data.frame(tag=dataset$tag, movieId=dataset$movieId)
	if (includeUser) {
		newDataset$userId <- dataset$userId
	}
	for (feature in features) {
		newDataset[[feature$featureName]] <- feature$transform(dataset[[feature$colName]])
	}
	newDataset
}

getFeatureNames <- function(features) {
	featureNames <- character(length(features))
	for (i in 1:length(features)) {
		featureNames[i] <- features[[i]]$featureName
	}
	featureNames
}

scaleDataset <- function( datasetToScale, referenceDataset, featuresToScale) {
	newDataset <- datasetToScale
	for (feature in featuresToScale) {
		featureVals <- referenceDataset[[feature$featureName]]
		featureVals <- featureVals[!is.na(featureVals)]
		newDataset[[feature$featureName]] <- (datasetToScale[[feature$featureName]] - mean(featureVals)) / sd(featureVals) 
	}
	newDataset
} 

filterTags <- function(dataset, tags) {
	dataset[dataset$tag %in% tags,]
}

removeFeaturesFromSameGrouping <- function(features, removeFeatures) {
	newFeatures <- list()
	for (feature in features) {
		conflict <- FALSE
		if (!is.null(feature$group)) {		
			for (removeFeature in removeFeatures) {
				if (!is.null(removeFeature$group)) {
					if (removeFeature$group==feature$group) {
						conflict <- TRUE
						break
					}
				}
			}
		}	
		if (!conflict) {
			newFeatures[[length(newFeatures)+1]] <- feature
		}
	}
	newFeatures
}