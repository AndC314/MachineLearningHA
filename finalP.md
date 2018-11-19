---
title: "Accelerometer Fitness Performance"
output: html_document
---
# Pratical Machine Learning Assignment
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


# Summary
This document explores the training dataset given. Three machine learning models are compared and evaluated in term of accurancy for predicting the class of the exercise.

Random tree forest seems the best choice in this case with 0.9963 accuracy compared to Decision Tree : 0.7368 GBM : 0.9839.


```{r setup, echo=FALSE}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(
  fig.path = "figs/fig-"
)
```

```{r load, include=FALSE}
library(knitr, quiet=T)
library(caret, quiet=T)
library(rpart, quiet=T)
library(rpart.plot, quiet=T)
library(randomForest, quiet=T)
library(corrplot, quiet=T)
library(rattle, quiet=T)
set.seed(42)
```
## Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.


#Loading the datasets
We import the datasets and from the Training dataset we create a train/test partition with outcome variable "classe".

```{r}
train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")
inTrain  <- createDataPartition(train$classe, p=0.7, list=FALSE)
TrainSet <- train[inTrain, ]
TestSet  <- train[-inTrain, ]
```

# Exploratory Data Analysis
Let's check the TrainSet directly. Exploring this dataset saves a bit of execution time since it is 70% of the original one.
On the top of str command we find the number of observations and variables.
```{r}
str(TrainSet)
```
There is a lot of data in this dataset 19622 rows and 160 columns. The last column, 'classe', is the prediction we have to infer in this assignment. Below we can see the distribution of 'classe' with the user name. Every user has approximatively the same amount of classe.
```{r}
plot(x= train$user_name, y=train$classe)
```

We can use nearZeroVar to check for predictors which are almost constant.
```{r}
dim(TrainSet)
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]
rbind('Training set' = dim(TrainSet), 'Testing set'= dim(TestSet))
```
It was possible to narrow the variables down to 104. Again we apply some more cleaning in order to narrow it down more.

```{r cleaning}
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
dim(TrainSet)
```
Now we finally remove the first columns which have informations not needed in this assignment. We discard the time, name of subject, etc.
```{r}
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
rbind('Training set' = dim(TrainSet), 'testing set'= dim(TestSet))
```

We can finally check the correlation matrix for the Training Set cleaned.
```{r correlation}
corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.6, tl.col = rgb(0, 0, 0))
```


## Model
# Random Forest
This is random forest

```{r}
set.seed(42)
controlRF <- trainControl(method="cv", number=5, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf",trControl=controlRF)
modFitRandForest$finalModel
```

```{r}
predictRandForest <- predict(modFitRandForest, newdata=TestSet)
confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe)
confMatRandForest
```

```{r}
print(modFitRandForest)
plot(modFitRandForest)
```
```{r}
plot(varImp(modFitRandForest), top=20)
```

## Decision Trees

```{r}
set.seed(42)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
suppressWarnings(fancyRpartPlot(modFitDecTree))
```


```{r}
predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe)
confMatDecTree
```

```{r}
plot(confMatDecTree$table, col = "bisque", 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))
```


## Generalized Boosted Model
We first try the GBM with method 'cv' and no repeats. Then we check wether repeating increase significantly our predictions.

```{r}
set.seed(42)
controlGBM <- trainControl(method = "cv", number = 5)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
```

```{r}
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM
```

```{r}
set.seed(42)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
```

```{r}
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM
```
By increasing the repeats in the GBM there's an increase in accuracy from 0.9839 to 0.9886.
The accuracy of the 3 regression modeling methods above are:

Random Forest : 0.9963 Decision Tree : 0.7368 GBM (5 repeats) : 0.9886

# Conclusions


We try to solve a classification problem, then we must trie to use the classification method, at this time we sill use caret package: classification tree algorithm and random force. I also carried out 3-fold validation using the trainControl function.

Preparing Data
```{r}
training<-read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!"))
testing<-read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!"))
table(training$classe)
NA_Count = sapply(1:dim(training)[2],function(x)sum(is.na(training[,x])))
NA_list = which(NA_Count>0)
colnames(training[,c(1:7)])
training = training[,-NA_list]
training = training[,-c(1:7)]
training$classe = factor(training$classe)
testing = testing[,-NA_list]
testing = testing[,-c(1:7)]
```

```{r}
set.seed(42)
cv5 = trainControl(method="cv",number=5,allowParallel=TRUE,verboseIter=TRUE)
mod_rf = train(classe~., data=training, method="rf",trControl=cv5)
mod_tree = train(classe~.,data=training,method="rpart",trControl=cv5)
```
Verify performance:
```{r}
p_rf=predict(mod_rf,training)
p_tree=predict(mod_tree,training)
table(p_rf,training$classe); table(p_tree,training$classe)
```

```{r}
p_rf=predict(mod_rf,testing)
p_tree=predict(mod_tree,testing)
table(p_rf,p_tree)
```
From the results, it appears that the random forest model has the best accuracy for testing datas.

```{r}
answers=predict(mod_rf,testing)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers
```

