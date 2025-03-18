rm(list = ls())
diab.df=read.csv("C:/Users/my/Desktop/diabetes_binary_health_indicators_BRFSS2015.csv",stringsAsFactors=FALSE)
source('C:/Users/my/Desktop/Durumi/Week1/EDA/descriptive_analytics_utils.R')
head(diab.df)
str(diab.df)
attach(diab.df)
library(pastecs)
library(gmodels)
library(ggplot2)
library(gridExtra)

sum(is.na(diab.df))
sum(complete.cases(diab.df))

categorical.vars <- c('Diabetes_binary', 'HighBP', 'CholCheck','HighChol',
                      'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
                      'Fruits','Veggies', 'HvyAlcoholConsump','AnyHealthcare',
                      'NoDocbcCost','DiffWalk','Sex')
diab.df<- to.factors(diab.df,categorical.vars)
str(diab.df)
head(diab.df)

#################
## Categorical Variables ##
#################

table(diab.df$Diabetes_binary)
get.categorical.variable.stats(Diabetes_binary)
visualize.barchart(Diabetes_binary)

### High BP (Blood Pressure)
# 0 = no high BP, 1 = high BP
get.categorical.variable.stats(HighBP)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and HighBP.
# H1: Association exists between Diabetes_binary and HighBP.
fisher.test(Diabetes_binary,HighBP, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,HighBP)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and HighBP are associated.

### High Cholesterol
# 0 = no high cholesterol, 1 = high cholesterol
get.categorical.variable.stats(HighChol)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and HighChol.
# H1: Association exists between Diabetes_binary and HighChol.
fisher.test(Diabetes_binary,HighChol, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,HighChol)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and HighChol are associated.

### Cholesterol Check
# 0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years
get.categorical.variable.stats(CholCheck)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and CholCheck.
# H1: Association exists between Diabetes_binary and CholCheck.
fisher.test(Diabetes_binary,CholCheck, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,CholCheck)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and CholCheck are associated.

### Smoker
# Have you smoked at least 100 cigarettes in your lifetime?
# 0 = no, 1 = yes
get.categorical.variable.stats(Smoker)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and Smoker.
# H1: Association exists between Diabetes_binary and Smoker.
fisher.test(Diabetes_binary,Smoker, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,Smoker)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and Smoker are associated.

### Stroke
# 0 = no, 1 = yes
get.categorical.variable.stats(Stroke)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and Stroke.
# H1: Association exists between Diabetes_binary and Stroke.
fisher.test(Diabetes_binary,Stroke, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,Stroke)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and Stroke are associated.

### Heart Disease or Attack
# Coronary heart disease (CHD) or Myocardial infarction (MI)
# 0 = no, 1 = yes
get.categorical.variable.stats(HeartDiseaseorAttack)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and HeartDiseaseorAttack.
# H1: Association exists between Diabetes_binary and HeartDiseaseorAttack.
fisher.test(Diabetes_binary,HeartDiseaseorAttack, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,HeartDiseaseorAttack)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and HeartDiseaseorAttack are associated.

### Physical Activity
# Physical activity in past 30 days (excluding job-related activity)
# 1: Yes , 0: No
get.categorical.variable.stats(PhysActivity)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and PhysActivity.
# H1: Association exists between Diabetes_binary and PhysActivity.
fisher.test(Diabetes_binary,PhysActivity, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,PhysActivity)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and PhysActivity are associated.

### Fruits Consumption
# Consume fruit 1 or more times per day
# 1: Yes , 0: No
get.categorical.variable.stats(Fruits)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and Fruits.
# H1: Association exists between Diabetes_binary and Fruits.
fisher.test(Diabetes_binary,Fruits, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,Fruits)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and Fruits are associated.

### Vegetables Consumption
# Consume vegetables 1 or more times per day
# 1: Yes , 0: No
get.categorical.variable.stats(Veggies)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and Veggies.
# H1: Association exists between Diabetes_binary and Veggies.
fisher.test(Diabetes_binary,Veggies, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,Veggies)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and Veggies are associated.

### Heavy Alcohol Consumption
# Adult men >=14 drinks/week, adult women >=7 drinks/week
# 0 = no, 1 = yes
get.categorical.variable.stats(HvyAlcoholConsump)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and HvyAlcoholConsump.
# H1: Association exists between Diabetes_binary and HvyAlcoholConsump.
fisher.test(Diabetes_binary,HvyAlcoholConsump, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,HvyAlcoholConsump)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and HvyAlcoholConsump are associated.

### Any Healthcare Coverage
# 0 = no, 1 = yes
get.categorical.variable.stats(AnyHealthcare)

# Fisher & Chi-square test
# H0: No association between Diabetes_binary and AnyHealthcare.
# H1: Association exists between Diabetes_binary and AnyHealthcare.
fisher.test(Diabetes_binary,AnyHealthcare, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,AnyHealthcare)
# p-value < 0.05, reject null hypothesis.
# Therefore, Diabetes_binary and AnyHealthcare are associated.

```r
### No Doc bc Cost
# Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?
# 0 = no 1 = yes
get.categorical.variable.stats(NoDocbcCost)

# fisher & chisq.test 
#H0: There is no association between Diabetes_binary and NoDocbcCost.
#H1: There is an association between Diabetes_binary and NoDocbcCost.
fisher.test(Diabetes_binary,NoDocbcCost, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,NoDocbcCost)
# p-value < 0.05, rejecting the null hypothesis, indicating an association between the two variables.

### DiffWalk: Do you have serious difficulty walking or climbing stairs?
#0 = No, 1 = Yes
# Visualization
get.categorical.variable.stats(DiffWalk)

# Association test: fisher, chisq
# Testing association between Diabetes_binary and DiffWalk
fisher.test(Diabetes_binary, DiffWalk, simulate.p.value=TRUE)
chisq.test(Diabetes_binary, DiffWalk)
# p-value < 0.05, rejecting the null hypothesis, indicating an association between the two variables.

### Sex stats and bar chart
# 1: Male , 0: Female
get.categorical.variable.stats(Sex)
fisher.test(Diabetes_binary,Sex, simulate.p.value=TRUE)
chisq.test(Diabetes_binary,Sex)
# The p-values obtained through fisher & chisq tests are much smaller than the significance level of 0.05, so we reject the null hypothesis.
# Therefore, there is an association between Diabetes_binary and Sex.

# H0: There is no association between Diabetes_binary and Sex.
# H1: There is an association between Diabetes_binary and Sex.
## Thus, there is an association between Diabetes_binary and Sex. 
par(mfcol=c(1,1))

###################
## Continuous Variables ##
###################
library(car)
attach(diab.df)

# Box plot
head(diab.df)
qjadf <- diab.df[,c(5,15,16,17,20,21,22)]
boxplot(qjadf)

### GenHlth
# Would you say that in general your health is: scale 1-5
# 1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor

get.numeric.variable.stats(GenHlth)
sd(GenHlth)
# Histogram/Density plot
visualize.distribution(GenHlth)

# Box plot
visualize.boxplot(GenHlth, Diabetes_binary)

### Education
# 1: Never attended school or only kindergarten
# 2: Grades 1 through 8 (Elementary)
# 3: Grades 9 through 11 (Some high school)
# 4: Grade 12 or GED (High school graduate)
# 5: College 1 year to 3 years (Some college)
# 6: College 4 years or more (College graduate)
get.numeric.variable.stats(Education)
sd(Education)

# Histogram/Density plot
visualize.distribution(Education)

# Box plot
visualize.boxplot(Education, Diabetes_binary)

## BMI analysis 
# BMI: Body Mass Index
get.numeric.variable.stats(BMI)
sd(BMI)

# Histogram/Density plot
visualize.distribution(BMI)

# Box plot
visualize.boxplot(BMI, Diabetes_binary)

### Age
# 1(age 18 to 24) / 2(age 25 to 29)/ 3(age 30 to 34) / 4(age 35 to 39) / 5(age 40 to 44) / 6(age 45 to 49) / 7(age 50 to 54)
# 8(age 55 to 59) / 9(age 60 to 64)/ 10(age 65 to 69) / 11(age 70 to 74) / 12(age 75 to 79) / 13(age 80 or older) 
library(car)
attach(diab.df)
table(Age)
visualize.barchart(Age)
new.Age <- car::recode(Age,"1=1;2=1;3=2;4=2;5=3;6=3;7=4;8=4;9=5;10=5;11=6;12=6;13=7")
# The number of categories (13) was too many, so they were grouped into 7 categories.
get.categorical.variable.stats(new.Age)
visualize.barchart(new.Age)
diab.df$Age <- new.Age # Assigning new.Age to the original Age variable
attach(diab.df)

get.numeric.variable.stats(Age)
sd(Age)
# Histogram/Density plot
visualize.distribution(Age)

# Box plot
visualize.boxplot(Age, Diabetes_binary)

### Income
get.numeric.variable.stats(Income)
sd(Income)

# Histogram/Density plot
visualize.distribution(Income)

# Box plot
visualize.boxplot(Income, Diabetes_binary)

### MentHlth analysis
# MentHlth: Considering stress, depression, and emotional problems, 
# how many days in the past 30 days was your mental health not good? (1-30 days)

get.numeric.variable.stats(MentHlth)
sd(MentHlth)

# Histogram/Density plot
visualize.distribution(MentHlth)

# Box plot
visualize.boxplot(MentHlth, Diabetes_binary)

### PhysHlth analysis
# PhysHlth: Considering physical illness and injuries, 
# how many days in the past 30 days was your physical health not good? (1-30 days)
get.numeric.variable.stats(PhysHlth)
sd(PhysHlth)

# Histogram/Density plot
visualize.distribution(PhysHlth)

# Box plot
visualize.boxplot(PhysHlth, Diabetes_binary)

### GenHlth
# Would you say that in general your health is: scale 1-5
# 1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor

get.numeric.variable.stats(GenHlth)

# Histogram/Density plot
visualize.distribution(GenHlth)

# Box plot
visualize.boxplot(GenHlth, Diabetes_binary)

### Normalization of Continuous Variables
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# Normalize variables
numeric.vars <- c("Age","Income","BMI", "MentHlth", "PhysHlth", "GenHlth", "Education")
diab.df <- scale.features(diab.df, numeric.vars)
attach(diab.df)

# Normalized box plot
qjadf <- diab.df[,c(5,15,16,17,20,21,22)]
boxplot(qjadf)

##### Continuous-Continuous Variable Correlation ###############
diab.df_cor=cor(qjadf)
library(corrplot)
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(diab.df_cor, method="shade", shade.col=NA, tl.col="black", tl.srt=45,
         col=col(200), addCoef.col="black", order="AOE")
# Multicollinearity

#===============================================================
#=== Feature selection        ==================================
#===============================================================
library(caret)  # feature selection algorithm
library(randomForest) # random forest algorithm

# split data into training and test datasets in 60:40 ratio
# Use 60% of data for train, the rest for test
indexes <- sample(1:nrow(diab.df), size=0.6*nrow(diab.df))
train.data <- diab.df[indexes,]
test.data <- diab.df[-indexes,]

# rfe based feature selection algorithm
run.feature.selection <- function(num.iters=20, feature.vars, class.var){
  set.seed(10)
  variable.sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv", 
                        verbose = FALSE, returnResamp = "all", 
                        number = num.iters)
  results.rfe <- rfe(x = feature.vars, y = class.var, 
                     sizes = variable.sizes, 
                     rfeControl = control)
  return(results.rfe)
}
#===============================================================
#=== Logistic regression      ==================================
#===============================================================
library(caret) # model training and evaluation
library(ROCR) # model evaluation
source("C:/Users/my/Desktop/두루미/1주차/ModelComparison/performance_plot_utils.R") # plotting metric results

## separate feature and class variables
# Exclude the dependent variable, include the predictors
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

# build a logistic regression model
formula.init <- "Diabetes_binary ~ ."
formula.init <- as.formula(formula.init)
lr.model <- glm(formula=formula.init, data=train.data, family="binomial")

# view model details
summary(lr.model)

# perform and evaluate predictions
lr.predictions <- predict(lr.model, test.data, type="response")
lr.predictions <- as.factor(round(lr.predictions))
confusionMatrix(data=lr.predictions, reference=test.class.var, positive='1')


## glm specific feature selection
formula <- "Diabetes_binary ~ ."
formula <- as.formula(formula)
control <- trainControl(method="repeatedcv", number=10, repeats=2)
model <- train(formula, data=train.data, method="glm", 
               trControl=control)
importance <- varImp(model, scale=FALSE)
plot(importance)


# build new model with selected features
formula.new <- "Diabetes_binary ~ BMI + GenHlth + HighBP + Age + HighChol"
formula.new <- as.formula(formula.new)
lr.model.new <- glm(formula=formula.new, data=train.data, family="binomial")

# view model details
summary(lr.model.new)

# perform and evaluate predictions 
lr.predictions.new <- predict(lr.model.new, test.data, type="response") 
lr.predictions.new <- as.factor(round(lr.predictions.new)) ##
confusionMatrix(data=lr.predictions.new, reference=test.class.var, positive='1')

## model performance evaluations

# plot best model evaluation metric curves
lr.model.best <- lr.model.new
lr.prediction.values <- predict(lr.model.best, test.feature.vars, type="response")
predictions <- prediction(lr.prediction.values, test.class.var)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="LR ROC Curve")
plot.pr.curve(predictions, title.text="LR Precision/Recall Curve")

#===============================================================
#=== Decision tree classifier(DT) ==================================
#===============================================================
# split data into training and test datasets (60% of total for train)
indexes <- sample(1:nrow(diab.df), size=0.6*nrow(diab.df))
train.data <- diab.df[indexes,]
test.data <- diab.df[-indexes,]


library(rpart) # tree models 
library(caret) # feature selection
library(rpart.plot) # plot decision tree
library(ROCR) # model evaluation
library(e1071) # tuning model

## separate feature and class variables
# Create a dataset excluding the dependent variable
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

## build initial model with training data
# Build a model with all variables included
formula.init <- "Diabetes_binary ~ ."
# Convert "Diabetes_binary ~ ." into a formula
formula.init <- as.formula(formula.init)
# Set an appropriate cp value / minsplit: minimum number of observations
dt.model <- rpart(formula=formula.init, method="class", data=train.data, 
                  control = rpart.control(minsplit=10, cp=0.01))


## predict and evaluate results
# Predict using the created model (using test.feature.vars) - test.data with the dependent variable removed
# Predict the class each observation belongs to (type = "class")
dt.predictions <- predict(dt.model, test.feature.vars, type="class")
# Evaluate results
confusionMatrix(data=dt.predictions, reference=test.class.var, positive="1")


## dt specific feature selection
formula.init <- "Diabetes_binary ~ ."
formula.init <- as.formula(formula.init)
# Apply a consistent evaluation method for comparison
# Perform k-fold cross-validation repeated 2 times
# k-fold cross-validation: Divide the examples into 'number' of folds
control <- trainControl(method="repeatedcv", number=10, repeats=2)
# Train the predictive model (formula.init)
model <- train(formula.init, data=train.data, method="rpart", 
               trControl=control)
# Calculate variable importance
importance <- varImp(model, scale=FALSE)
# Visualization
plot(importance, cex.lab=0.5)


## build new model with selected features
# Based on the importance, select variables to create a new model
formula.new <- "Diabetes_binary ~ GenHlth + BMI + HighBP + DiffWalk+HighChol"
# Convert to formula
formula.new <- as.formula(formula.new)
dt.model.new <- rpart(formula=formula.new, method="class", data=train.data, 
                      control = rpart.control(minsplit=10, cp=0.01),
                      parms = list(prior = c(0.7, 0.3)))

## predict and evaluate results
# Predict the class each observation belongs to (type = "class")
dt.predictions.new <- predict(dt.model.new, test.feature.vars, type="class")
# Evaluate results
confusionMatrix(data=dt.predictions.new, reference=test.class.var, positive="1")
# Although the Accuracy value decreased,
# the Sensitivity, Detection Rate, Detection Prevalence, and Balanced Accuracy values increased.

# view model details
# dt.model.new is selected as the final model
dt.model.best <- dt.model.new
print(dt.model.best)
par(mfrow=c(1,1))
# Decision tree
prp(dt.model.new, type=1, extra=3, varlen=0, faclen=0)
# Including classification probabilities in the decision tree
library(partykit)
tree <- as.party(dt.model.new)
plot(tree)


## plot model evaluation metric curves
# Predict probabilities for each class (type = "prob")
dt.predictions.best <- predict(dt.model.best, test.feature.vars, type="prob")
dt.prediction.values <- dt.predictions.best[,2]
# Predict using test.class.var
predictions <- prediction(dt.prediction.values, test.class.var)
par(mfrow=c(1,2))
# ROC Curve (AUC: 0.74)
plot.roc.curve(predictions, title.text="DT ROC Curve")
plot.pr.curve(predictions, title.text="DT Precision/Recall Curve")

#===================================================
#=== NN classifier ==================================
#===================================================
# Sample 10,000 observations from the entire dataset
indexes <- sample(1:253680, 10000)
diab.df <- diab.df[indexes,]
indexes <- sample(1:nrow(diab.df), size=0.6*nrow(diab.df))
train.data <- diab.df[indexes,]
test.data <- diab.df[-indexes,]

library(caret)
library(ROCR)

# Transform the data (excluding the dependent variable)
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

# Data transformation
transformed.train <- train.data
transformed.test <- test.data
for (variable in categorical.vars){
  new.train.var <- make.names(train.data[[variable]])
  transformed.train[[variable]] <- new.train.var
  new.test.var <- make.names(test.data[[variable]])
  transformed.test[[variable]] <- new.test.var
}
transformed.train <- to.factors(df=transformed.train, variables=categorical.vars)
transformed.test <- to.factors(df=transformed.test, variables=categorical.vars)
transformed.test.feature.vars <- transformed.test[,-1]
transformed.test.class.var <- transformed.test[,1]

# Build model using training data
formula.init <- "Diabetes_binary ~ ."
formula.init <- as.formula(formula.init)
nn.model <- train(formula.init, data = transformed.train, method="nnet")

# Check the model
print(nn.model)

# Predict and evaluate results
nn.predictions <- predict(nn.model, transformed.test.feature.vars, type="raw")
confusionMatrix(data=nn.predictions, reference=transformed.test.class.var, 
                positive="X1")


# Select specific variables
formula.init <- "Diabetes_binary ~ ."
formula.init <- as.formula(formula.init)
control <- trainControl(method="repeatedcv", number=10, repeats=2)
model <- train(formula.init, data=transformed.train, method="nnet", 
               trControl=control)
importance <- varImp(model, scale=FALSE)
plot(importance, cex.lab=0.5)

## Select top 5 variables

# Build a new model with the selected variables
formula.new <- "Diabetes_binary ~CholCheck+HvyAlcoholConsump+BMI+HeartDiseaseorAttack+GenHlth"
formula.new <- as.formula(formula.new)
nn.model.new <- train(formula.new, data = transformed.train, method="nnet")

# Predict and evaluate results for the new model
nn.predictions.new <- predict(nn.model.new, transformed.test.feature.vars, type="raw")
confusionMatrix(data=nn.predictions.new, reference=transformed.test.class.var, 
                positive="X1")


# view hyperparameter optimizations
plot(nn.model.new, cex.lab=0.5)


# plot model evaluation metric curves
nn.model.best <- nn.model.new
nn.predictions.best <- predict(nn.model.best, transformed.test.feature.vars, type="prob")
nn.prediction.values <- nn.predictions.best[,2]
predictions <- prediction(nn.prediction.values, test.class.var)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="NN ROC Curve")
plot.pr.curve(predictions, title.text="NN Precision/Recall Curve")