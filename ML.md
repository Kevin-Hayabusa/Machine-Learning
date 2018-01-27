ML Project
================
Kevin
2018/1/14

``` r
knitr::opts_chunk$set(echo = TRUE)
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(randomForest)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(rpart)
```

Background
----------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??? a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

Data
----

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

What you should submit
----------------------

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Getting and cleaning data
-------------------------

``` r
myTraining <- read.csv(url("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
validation <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))
#number of class
table(myTraining$classe)
```

    ## 
    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

``` r
#clean up data and remove cols that mostly are NAs
naIndex <- vector()
for(i in 1:ncol(myTraining)){
        if(sum(is.na(myTraining[,i]))/nrow(myTraining)>0.95||sum(myTraining[,i]=="")/nrow(myTraining)>0.95)
                naIndex <- c(naIndex,i)
}
myTraining <- myTraining[,-naIndex]
#remove unnecessary cols
myTraining <- myTraining[,-c(1:7)]
#check the dim of dataset
dim(myTraining)
```

    ## [1] 19622    53

Data partition
--------------

70% of training will be used as training, 30% for testing the testing data will be used as validation

``` r
set.seed(9)
train <- createDataPartition(y=myTraining$classe, p=0.7, list=FALSE)
training <- myTraining[train, ]
testing <- myTraining[-train, ]
```

Princeple component analysis
----------------------------

Given that we have 86 variables even after trimming, we want to find a new set of variables that are uncorrelated and explain as much variance as possible

``` r
preProc <- preProcess(training[,-which(colnames(training)=="classe")],method="pca",thresh = 0.9)
trainingPC <- predict(preProc,training)
testingPC <- predict(preProc,testing)
```

Random forest
-------------

``` r
modFitRF <- randomForest(training$classe ~ .,   data=trainingPC)
predictions1 <- predict(modFitRF, testingPC, type = "class")
confusionMatrix(predictions1, testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1659   19    1    1    3
    ##          B    6 1106   20    4    6
    ##          C    5   13  992   41    5
    ##          D    3    0   13  914    6
    ##          E    1    1    0    4 1062
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9742          
    ##                  95% CI : (0.9698, 0.9781)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9673          
    ##  Mcnemar's Test P-Value : 1.44e-05        
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9910   0.9710   0.9669   0.9481   0.9815
    ## Specificity            0.9943   0.9924   0.9868   0.9955   0.9988
    ## Pos Pred Value         0.9857   0.9685   0.9394   0.9765   0.9944
    ## Neg Pred Value         0.9964   0.9930   0.9930   0.9899   0.9958
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2819   0.1879   0.1686   0.1553   0.1805
    ## Detection Prevalence   0.2860   0.1941   0.1794   0.1590   0.1815
    ## Balanced Accuracy      0.9927   0.9817   0.9768   0.9718   0.9901

Decision Tree
-------------

``` r
modFitDT <- rpart(classe ~ ., data=training, method="class")
predictions2 <- predict(modFitDT, testing, type = "class")
confusionMatrix(predictions2, testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1528  219   62  151   50
    ##          B   37  678   94   40   82
    ##          C   47   87  759  166  110
    ##          D   35   97   82  499   55
    ##          E   27   58   29  108  785
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.722           
    ##                  95% CI : (0.7104, 0.7334)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6458          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9128   0.5953   0.7398  0.51763   0.7255
    ## Specificity            0.8855   0.9467   0.9156  0.94534   0.9538
    ## Pos Pred Value         0.7602   0.7282   0.6493  0.64974   0.7795
    ## Neg Pred Value         0.9623   0.9069   0.9434  0.90913   0.9391
    ## Prevalence             0.2845   0.1935   0.1743  0.16381   0.1839
    ## Detection Rate         0.2596   0.1152   0.1290  0.08479   0.1334
    ## Detection Prevalence   0.3415   0.1582   0.1986  0.13050   0.1711
    ## Balanced Accuracy      0.8992   0.7710   0.8277  0.73149   0.8396

Random forest has a better accurcy

Validation
----------

``` r
#clean up data and remove cols that mostly are NAs
naIndex <- vector()
for(i in 1:ncol(validation)){
        if(sum(is.na(validation[,i]))/nrow(validation)>0.95||sum(validation[,i]=="")/nrow(validation)>0.95)
                naIndex <- c(naIndex,i)
}
validationCl <- validation[,-naIndex]
#remove unnecessary cols
validationCl <- validationCl[,-c(1:7)]
testdataPC <- predict(preProc,validationCl[,1:52])
validationCl$classe <- predict(modFitRF,testdataPC)
```
