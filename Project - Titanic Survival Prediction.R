#importing libraries
library(tidyverse)
library(scales)
library(caret)
library(dplyr)
library(purrr)
library(Hmisc)
library(rpart)
library(rpart.plot)
library(dummies)
library(ggplot2)
library(ROCR)
library(e1071)

set.seed(1)

#downloading titanic full data set from url and assign empty string or blank values to NA
titanic_data <- read.csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv",na.strings=c("","NA"))

#observe the structure of the data and its content
head(titanic_data)
str(titanic_data)
describe(titanic_data)

#change data type of some variables to factor
titanic_data$survived<-as.factor(titanic_data$survived)
titanic_data$pclass<-as.factor(titanic_data$pclass)
titanic_data$sex<-as.factor(titanic_data$sex)
titanic_data$embarked<-as.factor(titanic_data$embarked)

#plot pclass
titanic_data %>% ggplot(aes(pclass,fill=survived)) + geom_bar(aes(y = (..count..)/sum(..count..)), position="dodge") +        
  ylab("Rate %") + ggtitle("Survival Rate By Passenger Class") 

#plot sex
titanic_data %>% ggplot(aes(sex,fill=survived)) + geom_bar(aes(y = (..count..)/sum(..count..)), position="dodge") +        
  ylab("Rate %") + ggtitle("Survival Rate By Passenger Sex") 
titanic_data %>% ggplot(aes(pclass,fill=survived)) + geom_bar(aes(y = (..count..)/sum(..count..)),   position="dodge") + facet_wrap(~sex) +ylab("Rate %") +  ggtitle("Survival Rate By Passenger Class and Sex") 

#plot age before imputation
titanic_data %>% ggplot(aes(age, fill = survived))+
  geom_histogram(binwidth = 5, colour = "black", position = "dodge",alpha=1)+
  theme_bw()+ggtitle("Histogram - Age Distribution Of Passengers (Before Imputation)") +ylab("No of Passengers") 

titanic_data %>% ggplot(aes(age, fill = survived))+
  geom_histogram(binwidth = 5, colour = "black", position = "dodge",alpha=1)+
  theme_bw()+ ggtitle("Histogram - Age Distribution By Passenger Class (Before Imputation)") +facet_wrap(~pclass) +ylab("No of Passengers") 

#plot sibsp
titanic_data %>% ggplot(aes(sibsp,fill=survived)) + geom_bar(aes(y = (..count..)/sum(..count..)), position="dodge") +  ylab("Rate %") + ggtitle("Survival Rate By Number of Siblings/Spouses") 

#plot parch
titanic_data %>% ggplot(aes(parch,fill=survived)) + geom_bar(aes(y = (..count..)/sum(..count..)), position="dodge") + ylab("Rate %") + ggtitle("Survival Rate By Number of Parents/Children") +  scale_x_continuous(breaks = scales::pretty_breaks(15))

#plot fare
titanic_data %>% ggplot(aes(fare))+
  geom_histogram(binwidth = 5, colour = "black", position = "dodge",alpha=1)+
  theme_bw()+ ggtitle("Histogram - Fare Distribution") +ylab("No of Passengers") 

#plot embarked
titanic_data %>% ggplot(aes(embarked,fill=survived)) + geom_bar(aes(y = (..count..)/sum(..count..)), position="dodge") +  ylab("Rate %") + ggtitle("Survival Rate By Port of Embarkation") 

#get the mean values of age and fare by passenger class and store them as data frames
titanic_class_age_mean <- titanic_data %>% filter(!is.na(age)) %>% group_by(pclass) %>% dplyr::summarise(class_age_mean=mean(age)) %>% as.data.frame(.)
titanic_class_fare_mean <- titanic_data %>% filter(!is.na(fare)) %>% group_by(pclass) %>% dplyr::summarise(class_fare_mean=mean(fare)) %>% as.data.frame(.)

#function to calculate mode value 
Mode <- function(x) {
  u <- unique(x)
  u[which.max(tabulate(match(x, u)))]
}

#get mode value of embarked
mode_embarked <- Mode(titanic_data$embarked)

#join data frames obtained with the main titanic_data set 
titanic_data <- inner_join(titanic_data,titanic_class_age_mean,by="pclass") 
titanic_data <- inner_join(titanic_data,titanic_class_fare_mean,by="pclass") 

#replace the age, embarked and fare values with the logical condition for imputation
titanic_data <- mutate(titanic_data,age = if_else(is.na(age), class_age_mean, age), fare=if_else(is.na(fare), class_fare_mean, fare)) 
titanic_data <- mutate(titanic_data,embarked = if_else(is.na(embarked),mode_embarked,embarked)) 

#plot age after imputation
titanic_data %>% ggplot(aes(age, fill = survived))+
  geom_histogram(binwidth = 5, colour = "black", position = "dodge",alpha=1)+
  theme_bw()+ggtitle("Histogram - Age Distribution Of Passengers (After Imputation)") +ylab("No of Passengers") 

titanic_data %>% ggplot(aes(age, fill = survived))+
  geom_histogram(binwidth = 5, colour = "black", position = "dodge",alpha=1)+
  theme_bw()+ ggtitle("Histogram - Age Distribution By Passenger Class (After Imputation)") +facet_wrap(~pclass) +ylab("No of Passengers") 

#drop attributes that will not be used in the modelling
titanic_data<-titanic_data%>%select(-name,-class_age_mean,-class_fare_mean,-boat,-ticket,-body,-cabin,-home.dest)

#check the final full dataset before splitting to train and test set
head(titanic_data)
describe(titanic_data)

#split data into train and test dataset
test_index<-createDataPartition(y=titanic_data$survived,times=1,p=0.2,list=FALSE)
titanic_train<-titanic_data[-test_index,]
titanic_test<-titanic_data[test_index,]

#check the result of the split
dim(titanic_train)
dim(titanic_test)

#1st Model - Binomial Logistic Regression
glm_fit <- titanic_train %>% glm(survived ~ ., data=., family = "binomial")
summary(glm_fit)

#1st Model - Prediction and Accuracy
p_hat <- predict(glm_fit, newdata = titanic_test, type = "response")
y_hat <- ifelse(p_hat > 0.5, 1, 0) %>% factor
cm <- confusionMatrix(y_hat, titanic_test$survived)
cm
accuracy<-cm$overall['Accuracy']
accuracy

#1st Model - Performance Validation
#Plot ROC curve
predict_perf <- prediction(p_hat, titanic_test$survived)
metric <- performance(predict_perf, measure = "tpr", x.measure = "fpr")
plot(metric)
abline(a=0, b=1)
#Calculate AUC area
auc <- performance(predict_perf, measure = "auc")
auc

results<- data_frame(method = "Binomial Logistic Regression", Accuracy = accuracy, AUC=as.numeric(auc@y.values))

#2nd Model - Decision Tree 
tree <- rpart(survived ~ ., data=titanic_train, method="class")
rpart.plot(tree,type=4, cex=0.6,extra=101, box.palette="RdGn",branch.lty=3, shadow.col="gray", 
           nn=TRUE)

#2nd Model - Prediction and Accuracy
pred <- predict(tree, titanic_test,type="class")
confusionMatrix(pred, titanic_test$survived)

#2nd Model - Check Overfitting
pred_train <- predict(tree, titanic_train, type="class")
confusionMatrix(pred_train, titanic_train$survived)

#2nd Model - Cross Validation
set.seed(1)
folds = createMultiFolds(titanic_train$survived, k = 10, times = 3)
control <- trainControl(method = "repeatedcv", index = folds)
tree_cv <- train(survived ~ ., data = titanic_train, method = "rpart", trControl = control)
rpart.plot(tree_cv$finalModel,type=4, cex=0.6, extra=101, box.palette="RdGn",branch.lty=3, shadow.col="gray", nn=TRUE)

#2nd Model - Prediction and Accuracy After Cross Validation
pred_cv <- predict(tree_cv, titanic_test)
cm <- confusionMatrix(pred_cv, titanic_test$survived)
cm
accuracy<-cm$overall['Accuracy']
accuracy

#2nd Model - Check Overfitting Again
pred_cv_train <- predict(tree_cv, titanic_train)
confusionMatrix(pred_cv_train, titanic_train$survived)$overall['Accuracy']


#2nd Model - Performance Validation
#Plot ROC curve
pred_cv <- predict(tree_cv, titanic_test, type="prob")
predict_perf_cv <- prediction(pred_cv[,2], titanic_test$survived)
metric <- performance(predict_perf_cv, measure = "tpr", x.measure = "fpr")
plot(metric)
abline(a=0, b=1)
#Calculate AUC area
auc <- performance(predict_perf_cv, measure = "auc")
auc

results <- bind_rows(results, data_frame(method="Decision Tree", Accuracy = accuracy, AUC=as.numeric(auc@y.values)))
results
