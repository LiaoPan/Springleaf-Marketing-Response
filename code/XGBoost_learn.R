# This is getting a decent public LB score so I just want to note that I don't
# really know what I am doing. 

library(readr)
library(xgboost)

set.seed(616)

cat("reading the train and test data\n")
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")

feature.names <- names(train)[2:ncol(train)-1]  #get variables titles from header.
# names(train)  # 1934 variables

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))	#duplicate elements removed from sets.
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))	#access the column of train & numeralization.
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("replacing missing values with -1\n")
train[is.na(train)] <- -1
test[is.na(test)]   <- -1

# cat("sampling train to get around 8GB memory limitations\n")
# train <- train[sample(nrow(train), 40000),]
# gc()

# cat("training a XGBoost classifier\n")
# clf <- xgboost(data        = data.matrix(train[,feature.names]),
#               label       = train$target,
#               nrounds     = 100, # 100 is better than 200
#               objective   = "binary:logistic",
#               eval_metric = "auc")

# cat("making predictions in batches due to 8GB memory limitation\n")
# submission <- data.frame(ID=test$ID)
# submission$target1 <- NA 
# for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
#    submission[rows, "target1"] <- predict(clf, data.matrix(test[rows,feature.names]))
# }

cat("sampling train to get around 8GB memory limitations\n")
train <- train[sample(nrow(train), 80000),]
h <- sample(nrow(train), 40000)
train <-train[h,]
gc()   #relese memory.

cat("Making train and validation matrices\n")

dtrain <- xgb.DMatrix(data.matrix(train[,feature.names]), label=train$target)

val<-train[-h,]
gc()

dval <- xgb.DMatrix(data.matrix(val[,feature.names]), label=val$target)

watchlist <- watchlist <- list(eval = dval, train = dtrain)

param <- list(  objective           = "binary:logistic", 
                # booster = "gblinear",
                eta                 = 0.001,
                max_depth           = 13,  # changed from default of 6
                subsample           = 0.6,
                colsample_bytree    = 0.75,
                eval_metric         = "auc"
                # alpha = 0.0001, 
                # lambda = 1
                )

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 50, # changed from 300
                    verbose             = 2, 
                    early.stop.round    = 10,
                    watchlist           = watchlist,
                    maximize            = TRUE)

cat("making predictions in batches due to 8GB memory limitation\n")

submission <- data.frame(ID=test$ID)
submission$target <- NA 
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
    submission[rows, "target"] <- predict(clf, data.matrix(test[rows,feature.names]))

# submission2 <- data.frame(ID=test$ID)
# submission2$target2 <- NA 
# for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
#     submission2[rows, "target2"] <- predict(clf, data.matrix(test[rows,feature.names]))
#     
}


# submission <- merge(submission, submission2, by = "ID", all.x = TRUE)
# submission$target <- submission$target1*0.4 + submission$target2*0.6
# submission <- submission[,c(1,4)]

cat("saving the submission file\n")
write_csv(submission, "xgb_b6.csv")