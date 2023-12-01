Data <- read.csv("D:\\Second Yeaar\\Data_Minning\\train.csv")

names(Data)
View(head(Data))
colnames(Data)
n <- dim(New_dat)[1]
p <- dim(New_dat)[2]-1
names(New_dat)[70] <- "Y"


library(tidyverse)
library(caret)
library(leaps)
library(reshape2)
library(caret)
library(corrplot)
library(Hmisc)
library(stringr)
library(ggplot2)
library(hrbrthemes)
library(olsrr)
install.packages("hrbrthemes")
library(gbm)


#Data Exploration

hist(Data$critical_temp, 
     density = TRUE)
df <- data.frame(Data$critical_temp)

ggplot(Data, aes(x = critical_temp)) +
  geom_histogram(aes(y = ..density..),
                 colour = 1, fill = "white") +
  geom_density(lwd = 1, colour = 4,fill = 4,
               alpha = 0.25)

target <- ncol(Data)
features <- target - 1
feature_dat <- Data[,1:features]
target_dat <- as.data.frame(Data[,target])

# Feature selection (Stepwise)
model <- lm(critical_temp ~. , data = Data)

#Backwards Selection
k <- ols_step_backward_p(model, prem = 0.05)
#k$removed

#New data
New_dat <- select(Data,-k$removed)
dim(New_dat)
names(New_dat)

# Slpitting the data set(80%,20%)

set.seed(1)
randat <- sample(c(TRUE, FALSE),
        nrow(Data), replace=TRUE, 
        prob=c(0.8,0.2))
 train  <- Data[randat,]
 test   <- Data[!randat,]

targetvar <- ncol(New_dat)
features_var <- targetvar - 1
feature_train <- trainSet[,1:features_var]
feature_test <-  testSet[,1:features_var]
target_train <- trainSet[,targetvar]
target_test <- testSet[,targetvar]
target_train <- as.data.frame(target_train)


#Modelling

#1. Multiple_Regression
trainmodel <- lm(critical_temp ~., data = trainSet)
plot(trainmodel)
  
# Run algorithms using 10-fold cross validation

control <- trainControl(method="repeatedcv",
                       number=10, repeats=1)
 metricTarget <- "RMSE"
 set.seed(1)
 fit.lm <- train(critical_temp~., 
                 data=trainSet, method="lm",
                 metric=metricTarget, 
                 trControl=control)
 
##Training Error vs Test Error#######
set.seed(1)
 
 ntrain <-  round(n*0.8)
 train <-  sample(1:n, ntrain)
 train_MSE <-  rep(0, p)
 test_MSE <-  rep(0, p)
 
 for(i in 1:p){
   myfit <-  lm(Y ~ ., New_dat[train,
                      c(1:i, (p+1))])
   train_Y <-  New_dat[train, (p+1)]
   train_Y_pred <-  myfit$fitted
   train_MSE[i] <-  mean((train_Y - train_Y_pred)^2)
   
   test_Y <-  New_dat[-train, (p+1)]
   test_Y_pred <-  predict(myfit, 
                newdata = New_dat[-train, ])
   test_MSE[i] <-  mean((test_Y - test_Y_pred)^2)
 }
 
 ## type="n": don't plot; just set the plotting region
 plot(c(1, p), range(train_MSE, test_MSE), type="n", 
      xlab="# of variables", ylab="MSE",
      main = "Training Error vs Test Error")
 points(train_MSE, col = "blue", pch = 1)
 lines(train_MSE, col = "blue", pch = 1)
 points(test_MSE, col = "red", pch = 2)
 lines(test_MSE, col = "red", pch = 2)
 legend("topright", legend=c("Test Error", 
                        "Train Error"),
        col=c("red", "blue"), lty=1:2, cex=0.8)

 
# Making Predictions
 pred_lm <- predict(fit.lm, newdata=testSet)
 print(RMSE(pred_lm, target_test))
 print(R2(pred_lm, target_test))
#pred_actual <- as.data.frame( rbind(pred_lm,target_test))
 
# Scatter Plot 
plot(pred_lm, target_test,
     pch = 20,
     col = "blue",
     main = "Scatterplot for Observed vs
     Predicted critical Temperature",
     xlab = "Observed Critical Temperature",
     ylab = "Predicted critical Temperature ")
abline(lm(pred_lm~ target_test), col = "dark red")
 
#Gradient Boosting Method (GBM)

set.seed(1) 

fit.gbm <- train(critical_temp~., 
          data=trainSet, method="gbm", 
                  metric=metricTarget, 
          trControl=control, verbose=F)
 print(fit.gbm)
 
 #Making Prediction 
 pred_gbm <- predict(fit.gbm, newdata=testSet)
 print(RMSE(pred_gbm, target_test))
 print(R2(pred_gbm, target_test))
 
 # Scatter plot
 plot(pred_gbm, target_test,
      pch = 20,
      col = "black",
      main = "Scatterplot for Observed vs
      Predicted critical Temperature",
      xlab = "Observed Critical Temperature",
      ylab = "Predicted critical Temperature ")
 abline(lm(pred_gbm~ target_test), col = "dark red")
 
 #Compare baseline algorithms
 results <- resamples(list(MLR=fit.lm, GBM=fit.gbm))
 summary(results)
 dotplot(results)
 
 #Tunning for GBM 
 set.seed(1)
 grid <- expand.grid(n.trees = 400, 
                     interaction.depth = 16, 
                     shrinkage = 0.02, 
                     n.minobsinnode = 10)
 fit.final <- train(critical_temp~., 
                    data=trainSet, method="gbm", 
                    metric=metricTarget, 
                    tuneGrid=grid, trControl=control,
                    verbose=F)

 #Predictions on Testing dataset
 pred_gb_tuning <- predict(fit.final, newdata=testSet)
 print(RMSE(pred_gb_tuning, target_test))
 print(R2(pred_gb_tuning, target_test))
 #Plot
 plot(predictions, target_test,
      pch = 20,
      col = "black",
      main = "Scatterplot for Observed vs 
      Predicted critical Temperature",
      xlab = "Observed Critical Temperature",
      ylab = "Predicted critical Temperature ")
 abline(lm(predictions~ target_test), col = "dark red")
 
 