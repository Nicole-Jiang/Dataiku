#setwd("~/Documents/JOB!!/Interview/Dataiku/us_census_full")
##### Read Data #####
train.data= read.csv("census_income_learn.csv")
##### Exploratory Data Analysis #####
class(train.data[,42])  #Factor
summary(train.data[,42]) #187140- vs 12382+ # Unbalanced dependent variable!
names(train.data)[42]= "Saving"
str(train.data)
train.data$label= 0
train.data$label[train.data$Saving== levels(train.data$Saving)[2]] =1
train.data$label= as.factor(train.data$label)
str(train.data)
sample.index= sample(199522, 20000)
train.sample= train.data[sample.index,]
test.sample= train.data[-sample.index,]

##### Baseline Model: Random Forest. Classification, robust #####
library(randomForest) 
rf.fit= randomForest(train.sample$label~., data= train.sample[,1:41])
predict.rf= predict(rf.fit,test.sample[,1:41])
# Check overall accuracy
sum(predict.rf == test.sample$label)/length(predict.rf) #0.9405557
# Check confusion matrix
a= sum(predict.rf == 0 & test.sample$label== 0)
b= sum(predict.rf == 1 & test.sample$label== 0)
c= sum(predict.rf == 0 & test.sample$label== 1)
d= sum(predict.rf == 1 & test.sample$label== 1)
confusion.matrix.rf= matrix(c(a,c,b,d),ncol=2,nrow=2)
#Cohen's Kappa
p0= (a+d)/(a+b+c+d)
margin1= (a+b)/(a+b+c+d)*(a+c)
margin2= (c+d)/(a+b+c+d)*(b+d)
pe= (margin1 + margin2 )/ (a+b+c+d)
Kappa.rf= (p0-pe)/ (1-pe)
#Kappa =0.092


########## boost ########## 
library(rpart)
library(ada)
ada.fit = ada(train.sample$label~., data = train.sample[,1:41], type = 'discrete')
predict.ada= predict(ada.fit,test.sample[,1:41])
sum(predict.ada == test.sample$label)/length(predict.ada) #0.9495362
a= sum(predict.ada == 0 & test.sample$label== 0)
b= sum(predict.ada == 1 & test.sample$label== 0)
c= sum(predict.ada == 0 & test.sample$label== 1)
d= sum(predict.ada == 1 & test.sample$label== 1)
confusion.matrix.ada= matrix(c(a,c,b,d),ncol=2,nrow=2)
#Cohen's Kappa
p0= (a+d)/(a+b+c+d)
margin1= (a+b)/(a+b+c+d)*(a+c)
margin2= (c+d)/(a+b+c+d)*(b+d)
pe= (margin1 + margin2 )/ (a+b+c+d)
Kappa.ada= (p0-pe)/ (1-pe) #0.4507
# To do: Resampling Your Dataset, ensemble

########## resample #############
index1= which(train.data$label==1)
sample.index2= sample(c(1:199522,rep(index1,10)),10000)
train.sample2= train.data[sample.index2,]
test.sample2= train.data[-sample.index2,]
# random forest
rf.fit2= randomForest(train.sample2$label~., data= train.sample2[,1:41])
predict.rf2= predict(rf.fit2,test.sample2[,1:41])
# Check overall accuracy
sum(predict.rf2 == test.sample2$label)/length(predict.rf2) #0.95386

a= sum(predict.rf2 == 0 & test.sample2$label== 0)
b= sum(predict.rf2 == 1 & test.sample2$label== 0)
c= sum(predict.rf2 == 0 & test.sample2$label== 1)
d= sum(predict.rf2 == 1 & test.sample2$label== 1)
confusion.matrix.rf2= matrix(c(a,c,b,d),ncol=2,nrow=2)
#Cohen's Kappa
p0= (a+d)/(a+b+c+d)
margin1= (a+b)/(a+b+c+d)*(a+c)
margin2= (c+d)/(a+b+c+d)*(b+d)
pe= (margin1 + margin2 )/ (a+b+c+d)
Kappa.rf2= (p0-pe)/ (1-pe) #0.08568694

### ada boost ##
ada.fit2 = ada(train.sample2$label~., data = train.sample2[,1:41], type = 'discrete')
predict.ada2= predict(ada.fit2,test.sample2[,1:41])
sum(predict.ada2 == test.sample2$label)/length(predict.ada2) #0.876914861
a= sum(predict.ada2 == 0 & test.sample2$label== 0)
b= sum(predict.ada2 == 1 & test.sample2$label== 0)
c= sum(predict.ada2 == 0 & test.sample2$label== 1)
d= sum(predict.ada2 == 1 & test.sample2$label== 1)
confusion.matrix.ada2= matrix(c(a,c,b,d),ncol=2,nrow=2)
#Cohen's Kappa
p0= (a+d)/(a+b+c+d)
margin1= (a+b)/(a+b+c+d)*(a+c)
margin2= (c+d)/(a+b+c+d)*(b+d)
pe= (margin1 + margin2 )/ (a+b+c+d)
Kappa.ada2= (p0-pe)/ (1-pe) #0.3466

##### final #####
final.prediction =vector()
final.prediction[predict.rf == 1 | predict.ada==1]=1
final.prediction[predict.rf == 0 & predict.ada== 0]= 0
#final.prediction[predict.ada2 == 1 & predict.rf2== 0]= 0
sum(final.prediction == test.sample$label)/length(final.prediction) #0.8814861
a= sum(final.prediction == 0 & test.sample2$label== 0)
b= sum(final.prediction == 1 & test.sample2$label== 0)
c= sum(final.prediction == 0 & test.sample2$label== 1)
d= sum(final.prediction == 1 & test.sample2$label== 1)
confusion.matrix.f= matrix(c(a,c,b,d),ncol=2,nrow=2)
#Cohen's Kappa
p0= (a+d)/(a+b+c+d)
margin1= (a+b)/(a+b+c+d)*(a+c)
margin2= (c+d)/(a+b+c+d)*(b+d)
pe= (margin1 + margin2 )/ (a+b+c+d)
Kappa.ada2= (p0-pe)/ (1-pe) #0.3466