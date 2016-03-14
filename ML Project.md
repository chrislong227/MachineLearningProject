---
title: "Practical Machine Learning Course Project"
author: "Yang Long"
date: "March 13, 2016"
output: html_document
---
### Synopsis
#### Data Processing
Delete all the irrelevant variables as well as the variables with no non-missing value.

#### Model Construction
Our outcome variable is classe, a factor variable with 5 levels:

  - exactly according to the specification (Class A)
  - throwing the elbows to the front (Class B)
  - lifting the dumbbell only halfway (Class C)
  - lowering the dumbbell only halfway (Class D)
  - throwing the hips to the front (Class E)

Random forest algorithm is known for their ability of detecting the features that are important for classification. Also, the prediction evaluations will be based on maximizing the accuracy and minimizing the out-of-sample error. To that end, Random Forest model is used in this analysis. All available variables after cleaning will be used for prediction.

#### Cross-Validation
Cross-validation will be performed by subsampling our training data set randomly without replacement into 2 subsamples: sub training data (75% of the original training data set) and sub testing data (25%). Our models will be fitted on the sub training data set, and tested on the sub testing data. Once an appropriate model is choosen, it will be tested on the original testing data set.

#### Reproducibility
In order to make the analysis reproducible and the results replicable, the seed is set as 903 in this analysis.
```{r}
set.seed(903)
```

### Data Analysis
#### Data Processing
Load all the packages required in this analysis including caret and random forest:
```{r}
# install.packages("caret")
# install.packages("randomForest")
# install.packages("rpart")
# install.packages("lattice")
# install.packages("ggplot2")
library(caret)
library(randomForest)
```

Load the datasets:
```{r}
# If a value is missing in the dataset, we replace them with NAs.
# Read the testing set
test = read.csv("pml-testing.csv", na.strings = c("NA","#DIV/0!", ""))

# Read the training set
train = read.csv("pml-training.csv", na.strings = c("NA","#DIV/0!", ""))

# Delete columns with all missing values
test = test[ , colSums(is.na(test)) == 0]
train = train[ , colSums(is.na(train)) == 0]

# Also delete irrelevant variables
test = test[ , -c(1:7)]
train = train[ , -c(1:7)]

# Check the size of our datasets
dim(test)
dim(train)
```

Take a look at the classe variable distribution in the training dataset.
```{r}
plot = ggplot(data = train, aes(x = classe))
plot = plot + geom_bar(width = .5)
plot = plot + ggtitle("Bar Plot of levels of the variable classe within the training data set")
plot = plot + xlab("classe levels") + ylab("Frequency")

print(plot)
```

Subsample the training set to do cross-validation analysis. Here we make the sub training sample 75% the size as the original training sample, and the sub testing sample 25% the size as it:
```{r}
subSamp = createDataPartition(y = train$classe, p = 0.75, list = FALSE)

subTrain = train[subSamp, ] 
subTest = train[-subSamp, ]

dim(subTrain)
dim(subTest)
```

#### Feature Plot
```{r}
# plot features
total = which(grepl("^total", colnames(train), ignore.case = F))

totalAccel = train[, total]

featurePlot(x = totalAccel, y = train$classe, pch = 19, main = "Feature plot", plot = "pairs")
```

#### Predicting Model
Use random forest model to fit the dataset and use the resulting model to test the sub testing set:
```{r}
# Build the model
rfModel = randomForest(classe ~. , data = subTrain, method = "class")

# Predict the test data
pred = predict(rfModel, subTest, type = "class")

# Test results on sub testing data set:
confusionMatrix(pred, subTest$classe)
```

#### Out-of-Sample Error
Next step is to check much overfitting happens in our model:
```{r}
# predict on sub testing dataset
predTest = predict(rfModel, subTest)

# true accuracy of the predicted model
accuracy = sum(predTest == subTest$classe)/length(predTest)

# calculate the out-of-sample error
Error = 1 - accuracy
```

Thus our out-of-sample error is
```{r}
print(Error)
```

#### Results
The random forest model shows a valid model which we choose to use on the test dataset and the cross-validation result shows only a few mossclassification. 

### Prediction on Test Data
predict outcome levels on the original Testing data set using Random Forest algorithm:
```{r}
predict(rfModel, test, type = "class")
```
