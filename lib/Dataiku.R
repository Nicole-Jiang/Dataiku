#setwd("~/Documents/JOB!!/Interview/Dataiku/us_census_full")
##### Read Data #####
train.data= read.csv("census_income_learn.csv")
##### Exploratory Data Analysis #####
class(train.data[,42])  #Factor
summary(train.data[,42]) #187140- vs 12382+ # Unbalanced dependent variable!
names(train.data)[42]= "Saving"

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
confusion.matrix[1,1]= sum(predict.rf == levels(predict.rf)[2] & test.sample[,42] == test.sample[1,42])
confusion.matrix[1,2]= sum(predict.rf != test.sample[,42] & test.sample[,42] == test.sample[1,42])
confusion.matrix[2,1]= sum(predict.rf != test.sample[,42] & test.sample[,42] == test.sample[57,42])
confusion.matrix[2,2]= sum(predict.rf == test.sample[,42] & test.sample[,42] == test.sample[57,42])
#Cohen's Kappa
#Kappa high!




########## boost
library(rpart)
library(ada)
ada.fit = ada(train.sample$Saving~., data = train.sample, type = 'discrete')
predict.ada= predict(ada.fit,test.sample)
sum(predict.rf == test.sample[,42])/length(predict.rf)
confusion.matrix2= matrix(ncol=2,nrow=2)
confusion.matrix2[1,1]= sum(predict.rf == test.sample[,42] & test.sample[,42] == test.sample[1,42])
confusion.matrix2[1,2]= sum(predict.rf != test.sample[,42] & test.sample[,42] == test.sample[1,42])
confusion.matrix2[2,1]= sum(predict.rf != test.sample[,42] & test.sample[,42] == test.sample[64,42])
confusion.matrix2[2,2]= sum(predict.rf == test.sample[,42] & test.sample[,42] == test.sample[64,42])

# To do: Resampling Your Dataset, ensemble, penalty SVM
