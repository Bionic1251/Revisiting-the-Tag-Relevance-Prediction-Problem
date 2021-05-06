cross_val <- function(x,y,model,... ,ngroup=n){
    call <- match.call()
#    x <- as.matrix(x)
    n <- length(y)
    ngroup <- trunc(ngroup)
    if( ngroup < 2){
        stop ("ngroup should be greater than or equal to 2")
    }
    if(ngroup > n){
        stop ("ngroup should be less than or equal to the number of observations")
    }
  
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
    u <- vector("list",ngroup)
    cv.fit <- rep(NA,n)
    for(j in 1:ngroup){
        u <- model$fitFunction(x[-groups[[j]], ],y[-groups[[j]]],model$transformY, model$excludeVals,...)
        cv.fit[groups[[j]]] <-  model$predictFunction(u,x[groups[[j]],],model$untransformY,...)       
    }
  
    if(leave.out==1) groups <- NULL
    return(list(cv.fit=cv.fit, 
                ngroup=ngroup, 
                leave.out=leave.out,
                groups=groups, 
                call=call)) 
}

cross_val_group <- function(x,y,model, groupingValues, ... ,ngroup=n){
    call <- match.call()
#    x <- as.matrix(x)
	uniqueGroupingValues <- unique(groupingValues)
    n <- length(uniqueGroupingValues)
    ngroup <- trunc(ngroup)
    if( ngroup < 2){
        stop ("ngroup should be greater than or equal to 2")
    }
    if(ngroup > n){
        stop ("ngroup should be less than or equal to the number of observations")
    }
	  
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
    u <- vector("list",ngroup)
    cv.fit <- rep(NA,n)
    for(j in 1:ngroup){
    	groupingFieldValuesTrain <- uniqueGroupingValues[-groups[[j]]] 
    	groupingFieldValuesPredict <- uniqueGroupingValues[groups[[j]]] 
    	inTrainSet <- (groupingValues %in% groupingFieldValuesTrain)
    	inPredictSet <- (groupingValues %in% groupingFieldValuesPredict)
        u <- model$fitFunction(x[inTrainSet, ],y[inTrainSet],model$transformY,model$excludeVals, ...)
        cv.fit[inPredictSet] <-  model$predictFunction(u,x[inPredictSet,],model$untransformY,...)       
    }

    if(leave.out==1) groups <- NULL
    return(list(cv.fit=cv.fit, 
                ngroup=ngroup, 
                leave.out=leave.out,
                groups=groups, 
                call=call)) 
}
