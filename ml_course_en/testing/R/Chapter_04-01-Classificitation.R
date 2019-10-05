library(rpart) # Decision Trees
library(rpart.plot)
library(tidyverse)
library(class) # knn library
library(caret) # cross validation
library(mlbench) # hyperparameter tuning

#library(parallel)
library(rattle) 

library(doSNOW)

# install.packages("rattle")

# Load data
df <- read_csv("../../../data/telecom_churn.csv")

names(df) <- str_replace_all(names(df), " ", "_")
names(df) <- tolower(names(df))

# Summary statistics
summary(df)
glimpse(df)
df[,4:5] <- lapply(df[, 3:4], as.factor)
df[,4:5] <- lapply(df[, 3:4], as.numeric)
df$churn <- as.factor(df$churn)
df$churn <- ifelse(df$churn == "TRUE", "YES", "NO")
df$churn <- as.factor(df$churn)
summary(df)

state <- df$state
df$state <- NULL


# 70/30 split
set.seed(1)

indexes <- createDataPartition(df$churn, p = 0.7, list=F)
df.train <- df[indexes,]
df.test <- df[-indexes,]

prop.table(table(df.train$churn))
prop.table(table(df.test$churn))

rtree.model <- rpart(churn ~ ., df.train, control=rpart.control(maxdepth = 6), method = "class")
rtree.model

fancyRpartPlot(rtree.model)

predictandCM<- function(model, data)
{
  pred <-predict(model, data, type="class")
  confusionMatrix(table(data$churn, pred))
}
predictandCM(rtree.model,df.train)


glimpse(df.train)



ctrl <- trainControl(method="repeatedcv",
                     number = 10,
                     repeats = 3,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)


?caret::train

tree.grid <- expand.grid(maxdepth = seq(1,15))

rpart.fit2 <- caret::train(churn ~ .,  
                    data=df.train,
                    method = "rpart2", 
                    tuneGrid=tree.grid,
                    trControl=ctrl,
                    tuneLength = 15,
                    metric="ROC")

rpart.fit2
fancyRpartPlot(rpart.fit2$finalModel)

predictandCM(rpart.fit2$finalModel, df.train)
predictandCM(rpart.fit2$finalModel, df.test)



names(caret::getModelInfo())
caret::getModelInfo("knn")
modelLookup("knn")

?caret::train

str(df.train)

df.train.scaled <- df.train
df.train.scaled[,names(df.train.scaled) != "churn"] <-
  lapply(df.train.scaled[, names(df.train.scaled) != "churn"], scale)

knn.grid <- expand.grid(k = seq(1,10))
knn.fit <- caret::train(churn ~ .,
                        data=df.train.scaled,
                        method = "knn",
                        tuneGrid=knn.grid,
                        trControl=ctrl,
                        metric="ROC")
knn.fit
summary(knn.fit)
plot(knn.fit)

knn.fit$finalModel



# Do feature selection with the tree model

caretFuncs$summary <- defaultSummary

ctrl <- rfeControl(functions=caretFuncs, 
                   method = "repeatedcv",
                   repeats =3,
                   number = 10,
                   returnResamp="final", verbose = TRUE)

trainctrl <- trainControl(classProbs= TRUE,
                          summaryFunction = twoClassSummary)

str(df.train)


numberofcores = detectCores()  # review what number of cores does for your environment
numberofcores
cl <- makeCluster(numberofcores, type = "SOCK")
registerDoSNOW(cl)

system.time (
  tree.fit2 <- rfe(churn ~ ., 
                   df.train, 
                   rfeControl=ctrl,
                   method="rpart2",
                   metric = "Accuracy",
                   trControl = trainctrl)
  )

stopCluster(cl)
registerDoSEQ()

best.features <- tree.fit2$optVariables

ctrl <- trainControl(method="repeatedcv",
                     number = 10,
                     repeats = 3,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)


?caret::train
tree.grid <- expand.grid(maxdepth = seq(1,15))

rpart.fit3 <- caret::train(df.train[,best.features],
                           df.train$churn,
                           method = "rpart2", 
                           tuneGrid=tree.grid,
                           trControl=ctrl,
                           tuneLength = 15,
                           metric="ROC")

predictandCM(rpart.fit3$finalModel, df.test)
predictandCM(rpart.fit3$finalModel, df.train)

plot(rpart.fit3)
