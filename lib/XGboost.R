#setwd("~/Documents/JOB!!/Interview/Dataiku/us_census_full")
##### Read Data #####
train.data= read.csv("census_income_learn.csv")
##### Exploratory Data Analysis #####
class(train.data[,42])  #Factor
summary(train.data[,42]) #187140- vs 12382+ # Unbalanced dependent variable!
names(train.data)[42]= "Saving"
train.data$label= 0
train.data$label[train.data$Saving== levels(train.data$Saving)[2]] =1
train.data$label= as.factor(train.data$label)
str(train.data)
sample.index= sample(199522, 5000)
train.sample= train.data[sample.index,]
test.sample= train.data[-sample.index,]

## xgboost  #####
library(xgboost)
col= sapply(train.sample,is.numeric)
xg.train= train.sample[,col]
dtrain <- xgb.DMatrix(as.matrix(xg.train),label = as.numeric(train.sample$label)-1)
xg.fit= xgboost(dtrain,max.depth = 20,
                eta = 0.5, nthread = 5, nround = 5, objective = "binary:logistic")
xg.test= test.sample[,col]
predict.xg= predict(xg.fit, as.matrix(xg.test))
predict.xg[]
sum(predict.xg == test.sample$label)/length(predict.xg)  #bad behabior
