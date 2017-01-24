#setwd("~/Documents/JOB!!/Interview/Dataiku/us_census_full")
##### Read Data #####
train.data= read.csv("census_income_learn.csv")
##### Exploratory Data Analysis #####
class(train.data[,42])  #Factor
summary(train.data[,42]) #187140- vs 12382+ # Unbalanced dependent variable!
names(train.data)[42]= "Saving"
str(train.data)
##### Baseline Model: Random Forest. Classification, robust #####
library(randomForest) 
## Choose best mtry ##
sample.index= sample(199522, 1000)
train.sample= train.data[sample.index,]
test.sample= train.data[-sample.index,]
rf= randomForest(Saving~., data= train.sample)
predict.rf= predict(rf,test.sample)
# Check overall accuracy
sum(predict.rf == test.sample[,42])/length(predict.rf)
# Check confusion matrix
confusion.matrix= matrix(ncol=2,nrow=2)
confusion.matrix[1,1]= sum(predict.rf == levels(predict.rf)[1] & test.sample[,42] == levels(predict.rf)[1])
confusion.matrix[1,2]= sum(predict.rf == levels(predict.rf)[2] & test.sample[,42] == levels(predict.rf)[1])
confusion.matrix[2,1]= sum(predict.rf == levels(predict.rf)[1] & test.sample[,42] == levels(predict.rf)[2])
confusion.matrix[2,2]= sum(predict.rf == levels(predict.rf)[2] & test.sample[,42] == levels(predict.rf)[2])
#Cohen's Kappa
p0= (confusion.matrix[1,1]+confusion.matrix[2,2])/sum(confusion.matrix)
margin1= (confusion.matrix[1,1]+confusion.matrix[1,2])/sum(confusion.matrix)*(confusion.matrix[1,1]+confusion.matrix[2,1])
margin2= (confusion.matrix[2,1]+confusion.matrix[2,2])/sum(confusion.matrix)*(confusion.matrix[1,2]+confusion.matrix[2,2])
pe= (margin1 + margin2 )/ sum(confusion.matrix)
Kappa= (p0-pe)/ (1-pe)
#Kappa =0.1647331




########## boost
library(rpart)
library(ada)
ada.fit = ada(train.sample$Saving~., data = train.sample, type = 'discrete')
predict.ada= predict(ada.fit,test.sample)
sum(predict.ada == test.sample[,42])/length(predict.ada)
confusion.matrix2= matrix(ncol=2,nrow=2)
confusion.matrix2[1,1]= sum(predict.ada == levels(predict.ada)[1] & test.sample[,42] == levels(predict.ada)[1])
confusion.matrix2[1,2]= sum(predict.ada == levels(predict.ada)[2] & test.sample[,42] == levels(predict.ada)[1])
confusion.matrix2[2,1]= sum(predict.ada == levels(predict.ada)[1] & test.sample[,42] == levels(predict.ada)[2])
confusion.matrix2[2,2]= sum(predict.ada == levels(predict.ada)[2] & test.sample[,42] == levels(predict.ada)[2])
p0= (confusion.matrix2[1,1]+confusion.matrix2[2,2])/sum(confusion.matrix2)
margin1= (confusion.matrix2[1,1]+confusion.matrix2[1,2])/sum(confusion.matrix2)*(confusion.matrix2[1,1]+confusion.matrix2[2,1])
margin2= (confusion.matrix2[2,1]+confusion.matrix2[2,2])/sum(confusion.matrix2)*(confusion.matrix2[1,2]+confusion.matrix2[2,2])
pe= (margin1 + margin2 )/ sum(confusion.matrix2)
Kappa= (p0-pe)/ (1-pe) #0.3455662

### XG boost ###

col= sapply(train.sample,is.numeric)
a= train.sample[,col]
train.sample$Saving[train.sample$Saving== levels(predict.ada)[1]]=0
train.sample$label= 0
train.sample$label[train.sample$Saving== levels(predict.ada)[2]] =1
t= factor(as.character(train.sample$Saving),levels= c(0,1))
library(xgboost)
dtrain <- xgb.DMatrix(as.matrix(a),label = train.sample$Saving)
xg.fit= xgboost(dtrain,max.depth = 2,
                eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")


# To do: Resampling Your Dataset, ensemble
