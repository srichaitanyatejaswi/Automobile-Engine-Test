## load libraries
library(caret)
library(DMwR)
library(VIM)
library(randomForest)
library(ROCR)
library(MASS)
library(car)
library(xgboost)
library(mlr)
library(dplyr)
library(e1071)



## merge main data and additional data 


add_train <-  read.csv("Train_AdditionalData.csv")
main_train <- read.csv("Train.csv")

add_test <-  read.csv("Test_AdditionalData.csv")
main_test <- read.csv("Test.csv")


## new df with index
new_df <- data.frame(ID = main_train$ID)
new_df1 <- data.frame(ID = main_test$ID)

## A 
a <- data.frame(ID = add_train[,1],Apass = "A")
c <-  left_join(x = new_df,y = a,by = "ID")

a1 <- data.frame(ID = add_test[,1],Apass = "A")
c1 <-  left_join(x = new_df1,y = a1,by = "ID")


## B
b <- data.frame(ID = add_train[,2],Bpass = "B")
d <-  left_join(x = c,y = b,by = "ID")

b1 <- data.frame(ID = add_test[,2],Bpass = "B")
d1 <-  left_join(x = c1,y = b1,by = "ID")


## Create a new column pass that indicates which test the engine has passed
for(i in 1:nrow(d)){
  d[i,"pass"] <-  paste0(na.omit(d[i,2]),na.omit(d[i,3]))
}

for(i in 1:nrow(d1)){
  d1[i,"pass"] <-  paste0(na.omit(d1[i,2]),na.omit(d1[i,3]))
}

sum(table(d$pass)) ## 3156
sum(table(d1$pass)) ## 1053

## remove all except few variables
rm(list=setdiff(ls(), c("main_train","d","main_test","d1")))

train_data_new <- data.frame(main_train,pass = d$pass)
test_data_new <- data.frame(main_test,pass = d1$pass)

##
setwd("C:\\Users\\surat\\OneDrive\\1.Insofe\\000.phd\\final")
write.csv(train_data_new,"train_data_new.csv",row.names = F)
write.csv(test_data_new,"test_data_new.csv",row.names = F)


##########################################################################################


## load data with new column "pass"
train_new  <- read.csv("train_data_new.csv")
test_new <- read.csv("test_data_new.csv")

## view the struture of the data
str(train_new)
unique(train_new$Number.of.Cylinders)

## convert the "number.of.cylinders" column to categorical because there are only 3 unique values
train_new$Number.of.Cylinders <- as.factor(as.character(train_new$Number.of.Cylinders))
test_new$Number.of.Cylinders <- as.factor(as.character(test_new$Number.of.Cylinders))

## remove ID column which is not significant for train data
train_new <- train_new[,-1]
test_new1 <- test_new[,-1]

## check whether the levels in test and train data are same 
train_new1 <- train_new[,-1]
for(i in 1:ncol(train_new1)){
  print(length(levels(train_new1[,i])) == length(levels(test_new1[,i])))
}
## all levels are same in train and test except one column "main.bearing.type"

## remove the column "main.bearing.type" which has constant value in the test data
unique(test_new$main.bearing.type)
prop.table(table(train_new$main.bearing.type)) ## even in train 99.93 % is of same level 

train_new$main.bearing.type <- NULL
test_new$main.bearing.type <- NULL


## check for missing values column wise
sort(colSums(is.na(train_new)),decreasing = TRUE)
## in all the columns with missing values we have 158 missing records

## visualize the missing data to check for any pattern
# aggr(train_new,col = c("#222760","#f6e9e0"),cex.lab = 1.5,combined = F)
aggr(train_new,col = c("#222760","#ff9900"),cex.lab = 1.5,combined = F,cex.axis = 0.4)


## impute the missing values
## use nearest neighbours mode to impute the missing values
train_new_imputed <- knnImputation(train_new,meth = "mostfrequent") 
test_new_imputed <- knnImputation(test_new,meth = "mostfrequent")


## remove except
rm(list=setdiff(ls(), c("train_new_imputed","test_new_imputed")))

## split data into test and validation
set.seed(1)
train_rows <- createDataPartition(train_new_imputed$y, p = 0.7, list = F)
train <- train_new_imputed[train_rows, ]
validation <- train_new_imputed[-train_rows, ]

#################################################################################################
####################################### best model - RF #########################################
#################################################################################################
# Random Search for optimal mtry
control <- trainControl(method="repeatedcv", number=3, repeats=3, search="random")
set.seed(1)
mtry <- sqrt(ncol(train_new_imputed))
rf_random <- train(y~., data=train_new_imputed, method="rf", metric="Accuracy", tuneLength=10, trControl=control)
print(rf_random) ## optimal mtry is 3 

## build model with manual search
set.seed(1)
rf10 <- randomForest(y~pass+Peak.Power+Fuel.Type,data = train,ntree = 9,mtry = 3)
y_pred_train = predict(rf10,train)
y_pred_validation = predict(rf10, validation)
confusionMatrix(data = train$y,reference = y_pred_train,positive = "pass") 
confusionMatrix(data = validation$y,reference = y_pred_validation,positive = "pass")  ## 87.16

## The variables in formula and ntree was tuned using manual search
randomForest <- randomForest(y~.,data = train,ntree = 9,mtry = 3)
varImpPlot(randomForest) ## all 3 variables are very important and they are enough to classify 

## Visualizations

## relation between pass and target
g1 <- ggplot(train_new_imputed,mapping = aes(x = pass,fill = y))
g1 + geom_bar()

## relation between peak.power and target
g <- ggplot(train_new_imputed,mapping = aes(x = Peak.Power,fill = y))
g + geom_bar()

## relation between Fuel.Type and target
g2 <- ggplot(train_new_imputed,mapping = aes(x = Fuel.Type,fill = y))
g2 + geom_bar()


## submission
y_pred <-  predict(rf10,test_new_imputed)

submission <- data.frame(ID = test_new_imputed$ID,y = y_pred)
write.csv(submission,"submission_4.csv",row.names = F)



#################################################################################################
########################################### logistic ############################################
#################################################################################################

glm <- glm(y~.,data = train,family = "binomial")

#Plot the ROC curve using the extracted performance measures (TPR and FPR)
y_pred_train <-  predict(glm,train,type = "response")
pred <- prediction(y_pred_train, train$y)
perf <- performance(pred, measure="tpr", x.measure="fpr")

plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(pred, measure="auc")
auc <- perf_auc@y.values[[1]]
print(auc) ## 92.75

## predict and measure accuracy
y_pred_train <-  predict(glm,train,type = "response")
y_pred_train <-  ifelse(test = (y_pred_train > 0.15),"pass","fail")
y_pred_validation <-  predict(glm, validation,"response")
y_pred_validation <-  ifelse(test = (y_pred_validation > 0.15),"pass","fail")
confusionMatrix(data = train$y,reference = y_pred_train,positive = "pass") ## train accuracy 87.38
confusionMatrix(data = validation$y,reference = y_pred_validation,positive = "pass")  ## validation accuracy 87.28


############ improve the model further ########################

#Use vif to find any multi-collinearity
log_reg_vif = vif(glm)
log_reg_vif 
## all are below 10..So there is no multi-collinearity

## Step AIC
log_reg_step <-  stepAIC(glm, direction = "both")
summary(log_reg_step)

## After step AIC the significant levels are "Cylinder.deactivationYes","displacementlow","Max..Torquelow","passAB"
## The attributes that are significant are "Peak.Power","pass","Cylinder.deactivation","displacement"

## build model with updated formula

glm_new <- glm(y~Peak.Power+pass+Cylinder.deactivation+displacement,data = train,family = "binomial")

#Plot the ROC curve using the extracted performance measures (TPR and FPR)
y_pred_train <-  predict(glm_new,train,type = "response")
pred <- prediction(y_pred_train, train$y)
perf <- performance(pred, measure="tpr", x.measure="fpr")

plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(pred, measure="auc")
auc <- perf_auc@y.values[[1]]
print(auc) ## 89.82

## predict and measure accuracy
y_pred_train <-  predict(glm_new,train,type = "response")
y_pred_train <-  ifelse(test = (y_pred_train > 0.3),"pass","fail")
y_pred_validation <-  predict(glm_new, validation,"response")
y_pred_validation <-  ifelse(test = (y_pred_validation > 0.3),"pass","fail")
confusionMatrix(data = train$y,reference = y_pred_train,positive = "pass") ## train accuracy 87.11
confusionMatrix(data = validation$y,reference = y_pred_validation,positive = "pass")  ## validation accuracy 87.16



########################################### SVM #################################################
#################################################################################################

set.seed(1)
svm <- svm(y~pass+Peak.Power+Fuel.Type+displacement+piston.type,data = train,kernel = "radial",gamma = 10)

y_pred_train <-  predict(svm,train)
y_pred_validation <-  predict(svm, validation)
confusionMatrix(data = train$y,reference = y_pred_train,positive = "pass") 
confusionMatrix(data = validation$y,reference = y_pred_validation,positive = "pass")  

## linear 87.36
## radial 87.36
## sigmoid 87.26
## polynomial degree 3 85.29
## radial gamma 10 87.46
## radial cost = 0.5 87.16


################################# XG boost #################################
############################################################################

## create n-1 dummy variables
dummy_train_x <- as.matrix(createDummyFeatures(train[,-1],method = "reference"))
dummy_train_y <- ifelse(test = (train$y == "pass"),yes = 1,no = 0)

dummy_validation_x <- as.matrix(createDummyFeatures(validation[,-1],method = "reference"))
dummy_validation_y <- ifelse(test = (validation$y == "pass"),yes = 1,no = 0)


## fit the model
xgb <- xgboost(data = dummy_train_x, label = dummy_train_y,objective = "binary:logistic",nrounds = 4)

pred_train <- predict(xgb, dummy_train_x)
pred_train_class <- as.numeric(pred_train > 0.5)
pred_train <- ifelse(test = (pred_train_class == 1),yes = "pass",no = "fail")

pred_validation <- predict(xgb,dummy_validation_x)
pred_validation_class <- as.numeric(pred_validation > 0.5)
pred_validation <- ifelse(test = (pred_validation_class == 1),yes = "pass",no = "fail")

confusionMatrix(data = pred_train,reference = train$y,positive = "pass") ## train accuracy 87.9
confusionMatrix(data = pred_validation,reference = validation$y,positive = "pass") ## validation accuracy 86.95




 













