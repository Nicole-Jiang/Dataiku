---
title: "Dataiku Test Report"
output:
  html_document: default
  html_notebook: default
---
# 1. Overview
This is a project from Dataiku. The main purpose of this project is to predict whether people save more or less than $50,000 / year. Some census infomation is give (see the "data" folder), such as age, class of worker, industry and so on. 

This document is orgenized as follows: 

+ First, a general explore of the train data set, such as dimension, head, response variables and missing values, is presented.
+ Second, Exploratory Data Analysis is conducted.
+ Third, different machine learning models are applied and evaluated.
+ Forth, the final prediction results are presented. 
+ Fifth, the problem and future improvements are discussed. 


# 2. Dataset overview
```{r, cache=T, warning= F, message= F}
##### Read Data #####
setwd("~/Documents/JOB!!/Interview/Dataiku/us_census_full")
train.data= read.csv("census_income_learn.csv")
data.colname=c("age", "class.of.worker", "industry.code", "occupation.code", "education", "wage.per.hour",
               "enrolled.in.edu.inst.last.wk", "marital.status", "major.industry.code", "major.occupation.code",
               "race", "hispanic.Origin", "sex", "member.of.a.labor.union", "reason.for.unemployment", 
               "full.or.part.time.employment.stat", "capital.gains", "capital.losses", 
               "federal.income.tax.liability", "tax.filer.status", "region.of.previous.residence", 
               "state.of.previous.residence", "detailed.household.and.family.stat", 
               "detailed.household.summary.in.household", "instance.weight", "migration.code.change.in.msa",
               "migration.code.change.in.reg", "migration.code.move.within.reg", "live.in.this.house.1.year.ago", 
               "migration.prev.res.in.sunbelt","num.persons.worked.for.employer", "total.person.earnings",
               "country.of.birth.father","country.of.birth.mother", "country.of.birth.self", "citizenship",
               "own.business.or.self.employed", "fill.inc.questionnaire.for.veterans.admin", "veterans.benefits",
               "weeks.worked.in.year", "year", "saving")#.Add.col.names
colnames(train.data)= data.colname
# Read package
library(ggplot2)
library(memisc)
library(randomForest)
library(ada)
library(glmnet)
# Explore the dataset
dim(train.data)
head(train.data)
summary(train.data$saving)
```

The dataset is imbalanced. So we should not only look at the overall prediction accuracy, but also the accuracy for each side. 

```{r,cache =T,warning= F, message= F}
missing_value= apply(train.data, 2, function(x) sum(x == " ?" | x == " Not in universe"))
miss_percentage= data.frame(x= data.colname[1:41], m=missing_value[1:41]/199522)
ggplot(data=miss_percentage,aes(x=x,y=m)) + geom_bar(stat="identity")+coord_flip() + ylim(0,1)

```
Here, we regard "?" and "Not in universe" as missing value. As we can see, there are a lot of missing values in some columns, some of which are almost zero, eg. "own business or self employed". However, the data was treated as a level of factors, we do not need to further manipulate them. 



# 2. EDA
We further explore some variable distributions and extreme values.

### 2.1 Age and marital status 
```{r,cache=T}
ggplot(data=data.frame(age= train.data$age, marital= train.data$marital),aes(x=age,fill=marital)) + geom_bar()
```

According to this bar plot, the number of people who is 90 years old is extremely high. After checking with the original data, I believe that this should be people who is equal or orlder than 90 years old. 
From about 20 years old, the married and divorced numbers are increasing. From 50 years old, the widowed number is increasing. 

### 2.2 Education
```{r, cache=T}

ggplot(data= data.frame(edu=train.data$education),aes(edu))+ geom_bar() + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

```

In this chart, we can see that "Children" and "High school graduate" are the most, followed by "Some college but no degree" and "Bechalors degree". This matches our common sense. 

### 2.3 Working status related info
Since there are too many missing values in "Wage per hour", "capital gains", "capital losses" and "federal income tax liability", I would not plot them. 
The following plot is base on "Class of worker" and "Weeks worked in year"

```{r, cache=T}
weeks = vector()
weeks[train.data$weeks == 0] = "0 week"
weeks[train.data$weeks <= 15 & train.data$weeks >0]= "less than 15 weeks"
weeks[train.data$weeks <= 30 & train.data$weeks >15] = "less than 30 weeks"
weeks[train.data$weeks <= 50 & train.data$weeks >30]= "less than 50 weeks"
weeks[train.data$weeks >=51] = "more than 51 weeks"
ggplot(data.frame(worker= train.data$class, week= weeks), aes(x = worker, fill = week)) + geom_bar(width = 0.9, position = "fill") +theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

```

Those who work more than 51 weeks are majority, in most classes of worker.



# 3. Modeling and performance evaluation

### 3.1 Data manipulation
I changed the response variable from "- 50000 " and "50000+" to 0 and 1.
```{r, cache=T, warning= F, message= F}
train.data$label= 0
train.data$label[train.data$saving== levels(train.data$saving)[2]] =1
train.data$label= as.factor(train.data$label)
```

### 3.2 Fit Random forest, Adaboost and Logistic regression model (resample)
I seperated dataset into train set and test set, to see the different model performance. (Since the observation is large and I'm using cross-validation, I only select 50000 observations each time.)
Also, I replicated observations with response variabl= 1 for 10 times. By doing this, the 0/1 are balanced and we can use logistic regression.
```{r, cache = T, warning= F, message= F}
####### Sample data ######
confusion.matrix= matrix(nrow= 4, ncol= 9)
for(i in 1:3){
  sample.index= sample(199522, 10000)
  train.sample= train.data[sample.index,]
  test.sample= train.data[-sample.index,]
  ##### Random forest #####
  rf.fit= randomForest(train.sample$label~., data= train.sample[,1:41])
  predict.rf= predict(rf.fit,test.sample[,1:41])
  sum(predict.rf == test.sample$label)/length(predict.rf)
  a= sum(predict.rf == 0 & test.sample$label== 0)
  b= sum(predict.rf == 1 & test.sample$label== 0)
  c= sum(predict.rf == 0 & test.sample$label== 1)
  d= sum(predict.rf == 1 & test.sample$label== 1)
  confusion.matrix[,(i*3-2)]= c(a,b,c,d)
  ##### Adaboost #####
  ada.fit = ada(train.sample$label~., data = train.sample[,1:41], type = 'discrete')
  predict.ada= predict(ada.fit,test.sample[,1:41])
  sum(predict.ada == test.sample$label)/length(predict.ada)
  a= sum(predict.ada == 0 & test.sample$label== 0)
  b= sum(predict.ada == 1 & test.sample$label== 0)
  c= sum(predict.ada == 0 & test.sample$label== 1)
  d= sum(predict.ada == 1 & test.sample$label== 1)
  confusion.matrix[,(i*3-1)]= c(a,b,c,d)
  ##### Logistic regression #####
  ## resample inbalanced data ##
  index1= which(train.data$label==1)
  sample.index2= sample(c(1:199522,rep(index1,10)),10000)
  train.sample2= train.data[sample.index2,]
  test.sample2= train.data[-sample.index2,]
  ## Choose only numerical data to fit logistic regression
  num.col= sapply(train.sample2, is.numeric)
  logi.fit= glm(train.sample2$label~., data = train.sample2[,num.col], family="binomial")
  summary(logi.fit)
  predict.logi= predict(logi.fit, test.sample2[,num.col], type="response")
  predict.logi[predict.logi > 0.5]=1
  predict.logi[predict.logi<=0.5]=0
  a= sum(predict.logi == 0 & test.sample2$label== 0)
  b= sum(predict.logi == 1 & test.sample2$label== 0)
  c= sum(predict.logi == 0 & test.sample2$label== 1)
  d= sum(predict.logi == 1 & test.sample2$label== 1)
  confusion.matrix[,i*3]= c(a,b,c,d)
}

```

### 3.3 Model performance
We can get confusion matrix from the test result (True Positive, True Negative, False Positive and False Negative)
The following shows the confusion matrix at the first iteration.
```{r,cache=T, warning= F, message= F}
matrix(confusion.matrix[,1],2,2, byrow= T)  # Random Forest
matrix(confusion.matrix[,2],2,2, byrow= T)  # Adaboost
matrix(confusion.matrix[,3],2,2, byrow= T)  # Logistic regression

```

+ Random forest has a low False Positive rate, but a high False Negative rate.
+ Adaboost has moderate FP and FN rate.
+ Logistic regression has a high FP rate, but a low FN rate. 

So, the model selection actually depend on the goal of prediction: Do we want to make correct predictions as many as possible? Do we want to identify certain category (saving more than 5000) as many as possible? Does loss occur when we make a wrong prediction?

Overall, these three models have advatages and disadvatages. Since I don't have enough information, I just pick Adaboost because it has the highest overall accuracy with moderate FP and FN rates. 




# 4. Prediction (with given test data)
Usually, after choosing the model, we should use all data in the training set to train the model. However, because of too many observations, this process would cause overflow issue. So I only ransomly choose 30000 obs (15%) to train the model.
```{r, cache =T, warning= F, message= F}
setwd("~/Documents/JOB!!/Interview/Dataiku/us_census_full")
test.data= read.csv("census_income_test.csv")
colnames(test.data)= data.colname
test.data$label= 0
test.data$label[test.data$saving== levels(test.data$saving)[2]] =1
test.data$label= as.factor(test.data$label)

#Fit the whole training set
sample.index3= sample(199522, 30000)
train.sample3= train.data[sample.index3,]
ada.fit = ada(train.sample3$label~., data = train.sample3[,1:41], type = 'discrete')
final.pred= predict(ada.fit, test.data[,1:41])

sum(predict.ada == test.sample$label)/length(predict.ada)

```

Predict accuracy is shown above.

```{r, warning= F, message= F}
a= sum(predict.ada == 0 & test.sample$label== 0)
b= sum(predict.ada == 1 & test.sample$label== 0)
c= sum(predict.ada == 0 & test.sample$label== 1)
d= sum(predict.ada == 1 & test.sample$label== 1)
confusion.matrix.ada= matrix(c(a,c,b,d),ncol=2,nrow=2)
confusion.matrix.ada

```
The confusion matrix shows that FN and FP is still large, but accaptable.

# 5. Problems and further improvement

### 5.1 Problems

+ Too many missing values
+ Some columns are unclear, eg "num persons worked for employer". Cannot figure out the internal relationship with other features. 

### 5.2 Further improvement

+ Feature engineering: Generate new features, using existing features, to make a better prediction. 
+ Apply better model, and tune the parameters.



