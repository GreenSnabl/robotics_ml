library(tidyverse)
library(nnet)
library(MASS)

df <- read_csv("SkillCraft1_Dataset.csv")
names(df)
attach(df)

df$TotalHours <- as.numeric(TotalHours)

df <- na.omit(df)

train <- sample(x=c(TRUE, FALSE), 
                size=nrow(df), 
                replace = T,
                p=c(0.7,0.3))

fit <- lda(LeagueIndex ~ APM + ActionLatency + ActionsInPAC, 
           data=df[train,])
pred <- predict(fit, df[!train,])

mean(pred$class == df$LeagueIndex[!train])

tab <- table(pred$class,df$LeagueIndex[!train])
tab
round(prop.table(tab),2)


df2 <- data.frame(x = df$APM[!train],
                  y = df$LeagueIndex[!train],
                  color = pred$class)

ggplot(data=df2, aes(x,y,color=color)) + 
  geom_point()

names(df)


fit.lr <- lm(APM ~ SelectByHotkeys + ActionsInPAC + MinimapRightClicks, df[!train,])
summary(fit.lr)


par(mfrow=c(2,2))
plot(fit.lr)



cor(df)
