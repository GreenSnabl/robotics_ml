library(tidyverse)
library(MASS)
library(boot)
library(caret)
library(car)
library(leaps)
library(mlbench)
# install.packages("mlbench")

df <- read_csv("SkillCraft1_Dataset.csv")
names(df)
attach(df)

cols <- c("TotalHours", "Age", "HoursPerWeek")
df <- df %>% mutate_at(cols, as.numeric)
head(df)

df <- apply(df, 2, function (x) ifelse(is.na(x), median(x, na.rm=T), x))
df <- data.frame(df)

summary(is.na(df))

test <- sample(1:nrow(df), size = nrow(df)/5)

df.test <- df[test,]
df.train <- df[-test,]

pre <- preProcess(df.train, method = c("center", "scale"))
df.train <- predict(pre, df.train)
summary(df.train)
var(df.train)

x.train <- df.train %>% dplyr::select(-APM)
y.train <- df.train$APM

rs <- regsubsets(x.train, y.train)
summary(rs)
?train
names(getModelInfo())
control <- trainControl(method="repeatedcv", number = 10, repeats = 5)
model <- train(APM ~ ., data=df.train, method="lm")
summary(model)
