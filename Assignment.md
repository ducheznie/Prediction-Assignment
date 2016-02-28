# Prediction Assignment Writeup
Thomas So  
February 28, 2016  

## Executive Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E)

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


## Data

The training data set is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

To start off, we will load the proper libraries, training, and testing data. 


```r
## setting seed and loading libraries
set.seed(12345)
library(caret)
library(ggplot2)

## loading data
TrainData <- read.csv("pml-training.csv", header = TRUE, na.strings=c("NA", "#DIV/0!"))
TestData <- read.csv("pml-testing.csv", header = TRUE, na.strings=c("NA", "#DIV/0!"))
```

## Cleaning and Processing

There are many NA values which needs to be removed, as well as variables that we believe have no affect on the prediction of how well an activity was performed.
The variables that will be removed have to do with user information and time, which are irrelevant to our prediction.


```r
## dimension before processing
dim(TrainData)
```

```
## [1] 19622   160
```


```r
## removing variables with at least one NA value
noNATrainData <- TrainData[, apply(TrainData, 2, function(x) !any(is.na(x)))]
dim(noNATrainData)
```

```
## [1] 19622    60
```


```r
## removing variables with user information and time
cleanTrainData <- noNATrainData[,-c(1:8)]
dim(cleanTrainData)
```

```
## [1] 19622    52
```

20 clean cases are set aside as test data set. 


```r
cleanTestData <- TestData[, names(cleanTrainData[,-52])]
dim(cleanTestData)
```

```
## [1] 20 51
```

## Data Partitioning and Prediction Process

We partition the cleaned data set to obtain a 75% training set and a 25% test set, which is indepedent of the 20 cases that were set aside above.


```r
## partitioning the data
inTrain <-createDataPartition(y = cleanTrainData$classe, p = 0.75, list = FALSE)
training <- cleanTrainData[inTrain,]
testing <- cleanTrainData[-inTrain,]
dim(training)
```

```
## [1] 14718    52
```


```r
dim(testing)
```

```
## [1] 4904   52
```

## Model: Random Forest

The first model we try is Random Forest.


```r
fitControl <- trainControl(method="cv", number=5, allowParallel=TRUE, verbose=TRUE)
rfFit <- train(classe ~. , data = training, method = "rf", trControl = fitControl, verbose = FALSE)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=26 
## - Fold1: mtry=26 
## + Fold1: mtry=51 
## - Fold1: mtry=51 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=26 
## - Fold2: mtry=26 
## + Fold2: mtry=51 
## - Fold2: mtry=51 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=26 
## - Fold3: mtry=26 
## + Fold3: mtry=51 
## - Fold3: mtry=51 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=26 
## - Fold4: mtry=26 
## + Fold4: mtry=51 
## - Fold4: mtry=51 
## + Fold5: mtry= 2 
## - Fold5: mtry= 2 
## + Fold5: mtry=26 
## - Fold5: mtry=26 
## + Fold5: mtry=51 
## - Fold5: mtry=51 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 26 on full training set
```


```r
rfFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, verbose = FALSE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 26
## 
##         OOB estimate of  error rate: 0.65%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4178    4    1    1    1 0.001672640
## B   17 2818   12    0    1 0.010533708
## C    0   13 2548    6    0 0.007401636
## D    0    0   26 2383    3 0.012023217
## E    0    0    5    6 2695 0.004065041
```


```r
rfPred <- predict(rfFit, newdata = testing)
confusionMatrix(rfPred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395   11    0    0    0
##          B    0  934    4    0    0
##          C    0    3  848    5    1
##          D    0    0    3  799    1
##          E    0    1    0    0  899
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9941         
##                  95% CI : (0.9915, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9925         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9842   0.9918   0.9938   0.9978
## Specificity            0.9969   0.9990   0.9978   0.9990   0.9998
## Pos Pred Value         0.9922   0.9957   0.9895   0.9950   0.9989
## Neg Pred Value         1.0000   0.9962   0.9983   0.9988   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1905   0.1729   0.1629   0.1833
## Detection Prevalence   0.2867   0.1913   0.1748   0.1637   0.1835
## Balanced Accuracy      0.9984   0.9916   0.9948   0.9964   0.9988
```

Here are the predictions on the 20 clean cases that were set aside.


```r
rfPred20 <- predict(rfFit, newdata = cleanTestData)
rfPred20
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Results

Random forest trees were generated for the training data using cross-validation. 
The algorithim gave an accuracy of 99% with a 95% Confidence Interval of [0.9908, 0.9955].
A Kappa value of 0.99 was also achieved.

