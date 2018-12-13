#protein solubility dataset
#preprocessing data
total=read.csv("solubility2.csv")
total[,1]<-NULL
total[,1]<-NULL
total[ ,1][total[ ,1]>100]<-100
total[ ,1]<-total[ ,1]/100
colnames(total)[1] <- "solubility"
total[,3]<-total[,1]
total[,1]<-NULL
x<-na.omit(total)
x[ ,1]<-as.character(x[ ,1])
length(x[ ,1])
library("protr")
x[ ,1] = x[ ,1][(sapply(x[ ,1], protcheck))]
length(x[ ,1])
x1 = t(sapply(x[ ,1], extractAAC))
x[ ,3:22] <- x1[ ,1:20]
x[ ,1]<-NULL
x[ ,22]<-x[ ,1]
x[ ,1]<-NULL
colnames(x)[21] <- "solubility"
write.csv(x,"cleandata_1.csv")

clean7=read.csv("cleandata_1.csv")
clean7[ ,1]<-NULL
clean7$solubility[clean7$solubility >= 0.5] <-1
clean7$solubility[clean7$solubility < 0.5] <- 0
# 70% training data, 10 fold cross validation, 30% test data
library('caret')
set.seed(5)
TrainingDataIndex <- createDataPartition(clean7[ ,21], p=0.7, list = FALSE)
train7 <- clean7[TrainingDataIndex,]
test7<- clean7[-TrainingDataIndex,]

#decison tree
# train accuracy
testdata<-train7 
testdata[ ,21]<-NULL
install.packages("party")
library(party)
#output.tree <- ctree(solubility ~ ., data=train7)
output.tree <- ctree(solubility ~ ., data=train7,controls = ctree_control(mincriterion = 0.1))
pred <- predict(output.tree, newdata = testdata)
prediction<-data.frame(pred)
prediction[,1][prediction[,1] >= 0.5] <-1
prediction[,1][prediction[,1] < 0.5] <- 0
cm = as.matrix(table(Actual = train7$solubility, Predicted = prediction[,1]))
sum(diag(cm))/length(train7$solubility)

plot(output.tree)
#prune tree
#0.1  249  0.8357532
#0.2  189  0.8212341
#0.3  147  0.8030853
#0.4  141  0.7999093
#0.5  121 0.7881125
#0.6  97  0.7767695
#0.7  59  0.7617967
#0.8  45  0.7459165
#0.9  39  0.7436479
#default mincriterion 0.95     29 nodes   accuracy 0.7345735
#mincriterion 0.98     25 nodes   accuracy 0.7214156
#0.99 25  0.7214156
#1 1 0.5494555

# cv accuracy
d1 = NULL
for (x in seq(0,1,0.1)) {
  d2 = NULL
  yourData<-train7[sample(nrow(train7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7 <- yourData[testIndexes, ]
    train_7 <- yourData[-testIndexes, ]
    testdata_7<-test_7 
    test_7[ ,21]<-NULL
    output.tree <- ctree(solubility ~ ., data=train_7,controls = ctree_control(mincriterion = x))
    pred <- predict(output.tree, newdata = test_7)
    prediction<-data.frame(pred)
    prediction[,1][prediction[,1] >= 0.5] <-1
    prediction[,1][prediction[,1] < 0.5] <- 0
    cm = as.matrix(table(Actual = testdata_7$solubility, Predicted = prediction[,1]))
    accuracy=sum(diag(cm))/length(testdata_7$solubility)
    d2 = rbind(d2, data.frame(i,accuracy))
  }
  average_accuracy=sum(d2[1:10,2])/10 
  d1 = rbind(d1, data.frame(x,average_accuracy))
}
write.csv(d1,"DT_1.csv")
#test data
testdata<-test7 
testdata[ ,21]<-NULL
d1 = NULL
for (x in seq(0,1,0.1)) {
  output.tree <- ctree(solubility ~ ., data=train7,controls = ctree_control(mincriterion = x))
  pred <- predict(output.tree, newdata = testdata)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = test7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(test7$solubility)
  d1 = rbind(d1, data.frame(x,accuracy))
}
write.csv(d1,"DT_2.csv")
# choose mincriterion 0.8
# change training data ratio 
#calculate train accuracy

d1 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  output.tree <- ctree(solubility ~ ., data=train_7,controls = ctree_control(mincriterion = 0.8))
  pred <- predict(output.tree, newdata = testdata)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = train_7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(train_7$solubility)
  d1 = rbind(d1, data.frame(x,accuracy))
}
write.csv(d1,"DT_3.csv")
#cv accuracy
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  d3 = NULL
  yourData<-train_7[sample(nrow(train_7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7data <- yourData[testIndexes, ]
    train_7data <- yourData[-testIndexes, ]
    testdata_7data<-test_7data 
    test_7data[ ,21]<-NULL
    output.tree <- ctree(solubility ~ ., data=train_7data,controls = ctree_control(mincriterion = 0.8))
    pred <- predict(output.tree, newdata = test_7data)
    prediction<-data.frame(pred)
    prediction[,1][prediction[,1] >= 0.5] <-1
    prediction[,1][prediction[,1] < 0.5] <- 0
    cm = as.matrix(table(Actual = testdata_7data$solubility, Predicted = prediction[,1]))
    accuracy=sum(diag(cm))/length(testdata_7data$solubility)
    d3 = rbind(d3, data.frame(i,accuracy))
  }
  average_accuracy=sum(d3[1:10,2])/10 
  d2 = rbind(d2, data.frame(x,average_accuracy))
}
d1$accuracy2=d2[,2]
write.csv(d1,"DT_3.csv")
# test data
d1 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  output.tree <- ctree(solubility ~ ., data=train_7,controls = ctree_control(mincriterion = 0.8))
  pred <- predict(output.tree, newdata = test7)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = test7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(test7$solubility)
  d1 = rbind(d1, data.frame(x,accuracy))
}

# svm 
# kernel  
#train accuracy
library("e1071")
testdata<-train7 
testdata[ ,21]<-NULL
svm_model <- svm(solubility ~ ., train7,kernel ="radial")
pred <- predict(svm_model, newdata = testdata)
prediction<-data.frame(pred)
prediction[,1][prediction[,1] >= 0.5] <-1
prediction[,1][prediction[,1] < 0.5] <- 0
cm = as.matrix(table(Actual = train7$solubility, Predicted = prediction[,1]))
sum(diag(cm))/length(train7$solubility)
#radial     0.8393829   cv  0.7363739
#linear     0.7068966      0.6996154
#sigmoid    0.4664247     0.5131201
#polynomial   0.7654265   0.647427
# kernel cv accuracy


#cv accuracy
d2 = NULL
yourData<-train7[sample(nrow(train7)),]
folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
for (i in seq(1,10,1)) {
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_7 <- yourData[testIndexes, ]
  train_7 <- yourData[-testIndexes, ]
  testdata_7<-test_7 
  test_7[ ,21]<-NULL
  svm_model <- svm(solubility ~ ., train_7,kernel ="polynomial")
  pred <- predict(svm_model, newdata = test_7)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = testdata_7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(testdata_7$solubility)
  d2 = rbind(d2, data.frame(i,accuracy))
}
average_accuracy=sum(d2[1:10,2])/10 
average_accuracy

# training data ratio
# train accuracy
d1 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  svm_model <- svm(solubility ~ ., train_7,kernel ="radial")
  pred <- predict(svm_model, newdata = testdata)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = train_7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(train_7$solubility)
  d1 = rbind(d1, data.frame(x,accuracy))
}
# cv accuracy
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  d3 = NULL
  yourData<-train_7[sample(nrow(train_7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7data <- yourData[testIndexes, ]
    train_7data <- yourData[-testIndexes, ]
    testdata_7data<-test_7data 
    test_7data[ ,21]<-NULL
    svm_model <- svm(solubility ~ ., train_7data,kernel ="radial")
    pred <- predict(svm_model, newdata = test_7data)
    prediction<-data.frame(pred)
    prediction[,1][prediction[,1] >= 0.5] <-1
    prediction[,1][prediction[,1] < 0.5] <- 0
    cm = as.matrix(table(Actual = testdata_7data$solubility, Predicted = prediction[,1]))
    accuracy=sum(diag(cm))/length(testdata_7data$solubility)
    d3 = rbind(d3, data.frame(i,accuracy))
  }
  average_accuracy=sum(d3[1:10,2])/10 
  d2 = rbind(d2, data.frame(x,average_accuracy))
}
d1$accuracy2=d2[,2]
write.csv(d1,"DT_4.csv")
#test data
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  svm_model <- svm(solubility ~ ., train_7,kernel ="radial")
  pred <- predict(svm_model, newdata = test7)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = test7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(test7$solubility)
  d2 = rbind(d2, data.frame(x,accuracy))
}
d1$SVM=d2[,2]
#boosted 

install.packages("adabag")
library("adabag")

train7f=train7
train7f$solubility<-as.factor(train7f$solubility)
testdata<-train7f 
testdata[ ,21]<-NULL
control=rpart.control(maxdepth=8)
adaboost = boosting(solubility~., data=train7f, mfinal=100, control=control)
adaboost.pred <- predict.boosting(adaboost,newdata=testdata)
cm = as.matrix(table(Actual = train7f$solubility, Predicted = adaboost.pred$class))
sum(diag(cm))/length(train7f$solubility)
#8  100  0.9909256      0.7150494
#8  50   0.9314882      0.7173427
#8  10   0.8203267      0.7286508
# 8   5  0.7912886      0.7164562 
# 10  5  0.7899274      0.7037392
# 6  5  0.7885662      0.7027849
# 4  5  0.7513612       0.7200555
# maxdepth 5 iteration 5     0.7749546    0.70645
#no control 0.7935572 

# cv accuracy
d2 = NULL
yourData<-train7f[sample(nrow(train7f)),]
folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
for (i in seq(1,10,1)) {
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_7 <- yourData[testIndexes, ]
  train_7 <- yourData[-testIndexes, ]
  testdata_7<-test_7 
  test_7[ ,21]<-NULL
  control=rpart.control(maxdepth=8)
  adaboost = boosting(solubility~., data=train_7, mfinal=100, control=control)
  adaboost.pred <- predict.boosting(adaboost,newdata=test_7)
  cm = as.matrix(table(Actual = testdata_7$solubility, Predicted = adaboost.pred$class))
  accuracy=sum(diag(cm))/length(testdata_7$solubility)
  d2 = rbind(d2, data.frame(i,accuracy))
}
average_accuracy=sum(d2[1:10,2])/10 
average_accuracy

# traning data ratio
# train accuracy
d1 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7f[ ,21], p=x, list = FALSE)
  train_7 <- train7f[TrainingDataIndex,]
  test_7<- train7f[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  control=rpart.control(maxdepth=8)
  adaboost = boosting(solubility~., data=train_7, mfinal=10, control=control)
  adaboost.pred <- predict.boosting(adaboost,newdata=testdata)
  cm = as.matrix(table(Actual = train_7$solubility, Predicted = adaboost.pred$class))
  accuracy=sum(diag(cm))/length(train_7$solubility)
  d1 = rbind(d1, data.frame(x,accuracy))
}


# cv accuracy
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7f[ ,21], p=x, list = FALSE)
  train_7 <- train7f[TrainingDataIndex,]
  test_7<- train7f[-TrainingDataIndex,]
  d3 = NULL
  yourData<-train_7[sample(nrow(train_7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7data <- yourData[testIndexes, ]
    train_7data <- yourData[-testIndexes, ]
    testdata_7data<-test_7data 
    test_7data[ ,21]<-NULL
    control=rpart.control(maxdepth=8)
    adaboost = boosting(solubility~., data=train_7data, mfinal=10, control=control)
    adaboost.pred <- predict.boosting(adaboost,newdata=test_7data)
    cm = as.matrix(table(Actual = testdata_7data$solubility, Predicted = adaboost.pred$class))
    accuracy=sum(diag(cm))/length(testdata_7data$solubility)
    d3 = rbind(d3, data.frame(i,accuracy))
  }
  average_accuracy=sum(d3[1:10,2])/10 
  d2 = rbind(d2, data.frame(x,average_accuracy))
}
d1$accuracy2=d2[,2]
write.csv(d1,"DT_5.csv")
#test data
test7f=test7
test7f$solubility<-as.factor(test7f$solubility)
d3 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7f[ ,21], p=x, list = FALSE)
  train_7 <- train7f[TrainingDataIndex,]
  test_7<- train7f[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  control=rpart.control(maxdepth=8)
  adaboost = boosting(solubility~., data=train_7, mfinal=10, control=control)
  adaboost.pred <- predict.boosting(adaboost,newdata=test7f)
  cm = as.matrix(table(Actual = test7f$solubility, Predicted = adaboost.pred$class))
  accuracy=sum(diag(cm))/length(test7f$solubility)
  d3 = rbind(d3, data.frame(x,accuracy))
}
d1$Boosting=d3[,2]

# knn
# change neigbor n 
# train accuracy
install.packages("class")
library("class")
knn.model=knn(train=train7[,1:20], test=train7[,1:20], cl=train7$solubility, k =30)
d=table(train7$solubility,knn.model)
accuracy=sum(diag(d))/sum(d)
accuracy
# k 100   accuracy  0.6515426   0.6465405
#  50   0.6705989       0.6565076
#  30   0.6873866       0.6601687
#   10  0.7295826       0.6660736
#    5  0.7717786      0.6655944
#  1    0.996824       0.62297

# cv accuracy 
d2 = NULL
yourData<-train7[sample(nrow(train7)),]
folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
for (i in seq(1,10,1)) {
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_7 <- yourData[testIndexes, ]
  train_7 <- yourData[-testIndexes, ]
  testdata_7<-test_7 
  test_7[ ,21]<-NULL
  knn.model=knn(train=train_7[,1:20], test=testdata_7[,1:20], cl=train_7$solubility, k =1)
  d=table(testdata_7$solubility,knn.model)
  accuracy=sum(diag(d))/sum(d)
  d2 = rbind(d2, data.frame(i,accuracy))
}
average_accuracy=sum(d2[1:10,2])/10 
average_accuracy

# traning data ratio
#train accuracy
d1 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  knn.model=knn(train=train_7[,1:20], test=train_7[,1:20], cl=train_7$solubility, k =10)
  d=table(train_7$solubility,knn.model)
  accuracy=sum(diag(d))/sum(d)
  d1 = rbind(d1, data.frame(x,accuracy))
}
# cv accuracy
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  d3 = NULL
  yourData<-train_7[sample(nrow(train_7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7data <- yourData[testIndexes, ]
    train_7data <- yourData[-testIndexes, ]
    testdata_7data<-test_7data 
    test_7data[ ,21]<-NULL
    knn.model=knn(train=train_7data[,1:20], test=testdata_7data[,1:20], cl=train_7data$solubility, k =10)
    d=table(testdata_7data$solubility,knn.model)
    accuracy=sum(diag(d))/sum(d)
    d3 = rbind(d3, data.frame(i,accuracy))
  }
  average_accuracy=sum(d3[1:10,2])/10 
  d2 = rbind(d2, data.frame(x,average_accuracy))
}
d1$accuracy2=d2[,2]
write.csv(d1,"DT_6.csv")

# test data
d4 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  knn.model=knn(train=train_7[,1:20], test=test7[,1:20], cl=train_7$solubility, k =10)
  d=table(test7$solubility,knn.model)
  accuracy=sum(diag(d))/sum(d)
  d4 = rbind(d4, data.frame(x,accuracy))
}
d1$KNN=d4[,2]
write.csv(d1,"DT_7.csv")


# bank dataset
# preprocessing
clean7=read.csv("bank.csv")
clean7[ ,1]<-NULL
colnames(clean7)[21]="solubility"
clean7$solubility=as.numeric(as.factor(clean7$solubility))
clean7$solubility[clean7$solubility == 1] <-0
clean7$solubility[clean7$solubility ==2] <- 1
# 70% training data, 10 fold cross validation, 30% test data
library('caret')
set.seed(5)
TrainingDataIndex <- createDataPartition(clean7[ ,21], p=0.7, list = FALSE)
train7 <- clean7[TrainingDataIndex,]
test7<- clean7[-TrainingDataIndex,]

# Decision tree
# train accuracy
testdata<-train7 
testdata[ ,21]<-NULL
install.packages("party")
library(party)
#output.tree <- ctree(solubility ~ ., data=train7)
output.tree <- ctree(solubility ~ ., data=train7,controls = ctree_control(mincriterion = 1))
pred <- predict(output.tree, newdata = testdata)
prediction<-data.frame(pred)
prediction[,1][prediction[,1] >= 0.5] <-1
prediction[,1][prediction[,1] < 0.5] <- 0
cm = as.matrix(table(Actual = train7$solubility, Predicted = prediction[,1]))
sum(diag(cm))/length(train7$solubility)

# cv accuracy
d1 = NULL
for (x in seq(0,1,0.1)) {
  d2 = NULL
  yourData<-train7[sample(nrow(train7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7 <- yourData[testIndexes, ]
    train_7 <- yourData[-testIndexes, ]
    testdata_7<-test_7 
    test_7[ ,21]<-NULL
    output.tree <- ctree(solubility ~ ., data=train_7,controls = ctree_control(mincriterion = x))
    pred <- predict(output.tree, newdata = test_7)
    prediction<-data.frame(pred)
    prediction[,1][prediction[,1] >= 0.5] <-1
    prediction[,1][prediction[,1] < 0.5] <- 0
    cm = as.matrix(table(Actual = testdata_7$solubility, Predicted = prediction[,1]))
    accuracy=sum(diag(cm))/length(testdata_7$solubility)
    d2 = rbind(d2, data.frame(i,accuracy))
  }
  average_accuracy=sum(d2[1:10,2])/10 
  d1 = rbind(d1, data.frame(x,average_accuracy))
}
write.csv(d1,"DT_8.csv")
# training data ratio
# parameter 0.7
# train accuracy
d1 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  output.tree <- ctree(solubility ~ ., data=train_7,controls = ctree_control(mincriterion = 0.7))
  pred <- predict(output.tree, newdata = testdata)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = train_7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(train_7$solubility)
  d1 = rbind(d1, data.frame(x,accuracy))
}
#cv accuracy
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  d3 = NULL
  yourData<-train_7[sample(nrow(train_7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7data <- yourData[testIndexes, ]
    train_7data <- yourData[-testIndexes, ]
    testdata_7data<-test_7data 
    test_7data[ ,21]<-NULL
    output.tree <- ctree(solubility ~ ., data=train_7data,controls = ctree_control(mincriterion = 0.7))
    pred <- predict(output.tree, newdata = test_7data)
    prediction<-data.frame(pred)
    prediction[,1][prediction[,1] >= 0.5] <-1
    prediction[,1][prediction[,1] < 0.5] <- 0
    cm = as.matrix(table(Actual = testdata_7data$solubility, Predicted = prediction[,1]))
    accuracy=sum(diag(cm))/length(testdata_7data$solubility)
    d3 = rbind(d3, data.frame(i,accuracy))
  }
  average_accuracy=sum(d3[1:10,2])/10 
  d2 = rbind(d2, data.frame(x,average_accuracy))
}
d1$accuracy2=d2[,2]
write.csv(d1,"DT_9.csv")
# test data
d10 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  output.tree <- ctree(solubility ~ ., data=train_7,controls = ctree_control(mincriterion = 0.7))
  pred <- predict(output.tree, newdata = test7)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = test7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(test7$solubility)
  d10 = rbind(d10, data.frame(x,accuracy))
}

#SVM
#kernel
#train accuracy
testdata<-train7 
testdata[ ,21]<-NULL
svm_model <- svm(solubility ~ ., train7,kernel ="radial")
pred <- predict(svm_model, newdata = testdata)
prediction<-data.frame(pred)
prediction[,1][prediction[,1] >= 0.5] <-1
prediction[,1][prediction[,1] < 0.5] <- 0
cm = as.matrix(table(Actual = train7$solubility, Predicted = prediction[,1]))
sum(diag(cm))/length(train7$solubility)
# kernel cv accuracy
#radial     0.9072789    0.9057521
#linear     0.8993711   0.8991396
#sigmoid    0.8061413    0.810627
#polynomial   0.9034406   0.9017752

#cv accuracy
d2 = NULL
yourData<-train7[sample(nrow(train7)),]
folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
for (i in seq(1,10,1)) {
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_7 <- yourData[testIndexes, ]
  train_7 <- yourData[-testIndexes, ]
  testdata_7<-test_7 
  test_7[ ,21]<-NULL
  svm_model <- svm(solubility ~ ., train_7,kernel ="radial")
  pred <- predict(svm_model, newdata = test_7)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = testdata_7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(testdata_7$solubility)
  d2 = rbind(d2, data.frame(i,accuracy))
}
average_accuracy=sum(d2[1:10,2])/10 
average_accuracy

# training data ratio
# train accuracy
d1 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  svm_model <- svm(solubility ~ ., train_7,kernel ="radial")
  pred <- predict(svm_model, newdata = testdata)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = train_7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(train_7$solubility)
  d1 = rbind(d1, data.frame(x,accuracy))
}
# cv accuracy
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  d3 = NULL
  yourData<-train_7[sample(nrow(train_7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7data <- yourData[testIndexes, ]
    train_7data <- yourData[-testIndexes, ]
    testdata_7data<-test_7data 
    test_7data[ ,21]<-NULL
    svm_model <- svm(solubility ~ ., train_7data,kernel ="radial")
    pred <- predict(svm_model, newdata = test_7data)
    prediction<-data.frame(pred)
    prediction[,1][prediction[,1] >= 0.5] <-1
    prediction[,1][prediction[,1] < 0.5] <- 0
    cm = as.matrix(table(Actual = testdata_7data$solubility, Predicted = prediction[,1]))
    accuracy=sum(diag(cm))/length(testdata_7data$solubility)
    d3 = rbind(d3, data.frame(i,accuracy))
  }
  average_accuracy=sum(d3[1:10,2])/10 
  d2 = rbind(d2, data.frame(x,average_accuracy))
}
d1$accuracy2=d2[,2]
write.csv(d1,"DT_10.csv")
#test data
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  svm_model <- svm(solubility ~ ., train_7,kernel ="radial")
  pred <- predict(svm_model, newdata = test7)
  prediction<-data.frame(pred)
  prediction[,1][prediction[,1] >= 0.5] <-1
  prediction[,1][prediction[,1] < 0.5] <- 0
  cm = as.matrix(table(Actual = test7$solubility, Predicted = prediction[,1]))
  accuracy=sum(diag(cm))/length(test7$solubility)
  d2 = rbind(d2, data.frame(x,accuracy))
}
d10$SVM=d2[,2]

#boosted 
library("adabag")
library("rpart")

train7f=train7
train7f$solubility<-as.factor(train7f$solubility)
testdata<-train7f 
testdata[ ,21]<-NULL
control=rpart.control(maxdepth=5)
adaboost = boosting(solubility~., data=train7f, mfinal=5, control=control)
adaboost.pred <- predict.boosting(adaboost,newdata=testdata)
cm = as.matrix(table(Actual = train7f$solubility, Predicted = adaboost.pred$class))
sum(diag(cm))/length(train7f$solubility)
#8  100  0.9216611       0.9125034
#8  50   0.9202275       0.9107935
#8  10   0.9164354       0.9110244
# 8   5   0.9145856      0.9078809
# 10  5   0.9083888      0.9088974
# 6  5     0.9153718     0.90802
# 4  5     0.913707      0.9086193
# maxdepth 5 iteration 5   0.913707  0.9097298


# cv accuracy
d2 = NULL
yourData<-train7f[sample(nrow(train7f)),]
folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
for (i in seq(1,10,1)) {
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_7 <- yourData[testIndexes, ]
  train_7 <- yourData[-testIndexes, ]
  testdata_7<-test_7 
  test_7[ ,21]<-NULL
  control=rpart.control(maxdepth=5)
  adaboost = boosting(solubility~., data=train_7, mfinal=5, control=control)
  adaboost.pred <- predict.boosting(adaboost,newdata=test_7)
  cm = as.matrix(table(Actual = testdata_7$solubility, Predicted = adaboost.pred$class))
  accuracy=sum(diag(cm))/length(testdata_7$solubility)
  d2 = rbind(d2, data.frame(i,accuracy))
}
average_accuracy=sum(d2[1:10,2])/10 
average_accuracy

# traning data ratio
# train accuracy
d1 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7f[ ,21], p=x, list = FALSE)
  train_7 <- train7f[TrainingDataIndex,]
  test_7<- train7f[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  control=rpart.control(maxdepth=8)
  adaboost = boosting(solubility~., data=train_7, mfinal=100, control=control)
  adaboost.pred <- predict.boosting(adaboost,newdata=testdata)
  cm = as.matrix(table(Actual = train_7$solubility, Predicted = adaboost.pred$class))
  accuracy=sum(diag(cm))/length(train_7$solubility)
  d1 = rbind(d1, data.frame(x,accuracy))
}


# cv accuracy
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7f[ ,21], p=x, list = FALSE)
  train_7 <- train7f[TrainingDataIndex,]
  test_7<- train7f[-TrainingDataIndex,]
  d3 = NULL
  yourData<-train_7[sample(nrow(train_7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7data <- yourData[testIndexes, ]
    train_7data <- yourData[-testIndexes, ]
    testdata_7data<-test_7data 
    test_7data[ ,21]<-NULL
    control=rpart.control(maxdepth=8)
    adaboost = boosting(solubility~., data=train_7data, mfinal=100, control=control)
    adaboost.pred <- predict.boosting(adaboost,newdata=test_7data)
    cm = as.matrix(table(Actual = testdata_7data$solubility, Predicted = adaboost.pred$class))
    accuracy=sum(diag(cm))/length(testdata_7data$solubility)
    d3 = rbind(d3, data.frame(i,accuracy))
  }
  average_accuracy=sum(d3[1:10,2])/10 
  d2 = rbind(d2, data.frame(x,average_accuracy))
}
d1$accuracy2=d2[,2]
write.csv(d1,"DT_12.csv")
#test data
test7f=test7
test7f$solubility<-as.factor(test7f$solubility)
d3 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7f[ ,21], p=x, list = FALSE)
  train_7 <- train7f[TrainingDataIndex,]
  test_7<- train7f[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  control=rpart.control(maxdepth=8)
  adaboost = boosting(solubility~., data=train_7, mfinal=100, control=control)
  adaboost.pred <- predict.boosting(adaboost,newdata=test7f)
  cm = as.matrix(table(Actual = test7f$solubility, Predicted = adaboost.pred$class))
  accuracy=sum(diag(cm))/length(test7f$solubility)
  d3 = rbind(d3, data.frame(x,accuracy))
}
d10$Boosting=d3[,2]
write.csv(d10,"DT_13.csv")

write.csv(d10,"DT_13.csv")
# KNN

clean7$job=as.numeric(as.factor(clean7$job))
clean7$marital=as.numeric(as.factor(clean7$marital))
clean7$education=as.numeric(as.factor(clean7$education))
clean7$default=as.numeric(as.factor(clean7$default))
clean7$housing=as.numeric(as.factor(clean7$housing))
clean7$loan=as.numeric(as.factor(clean7$loan))
clean7$contact=as.numeric(as.factor(clean7$contact))
clean7$month=as.numeric(as.factor(clean7$month))
clean7$day_of_week=as.numeric(as.factor(clean7$day_of_week))
clean7$poutcome=as.numeric(as.factor(clean7$poutcome))

str(clean7)

set.seed(5)
TrainingDataIndex <- createDataPartition(clean7[ ,21], p=0.7, list = FALSE)
train7 <- clean7[TrainingDataIndex,]
test7<- clean7[-TrainingDataIndex,]
# change neigbor n 
# train accuracy
install.packages("class")
library("class")
knn.model=knn(train=train7[,1:20], test=train7[,1:20], cl=train7$solubility, k =1)
d=table(train7$solubility,knn.model)
accuracy=sum(diag(d))/sum(d)
accuracy
# k 100   accuracy  0.9132908   0.9117189
#  50   0.9143082    0.9119495
#  30   0.916343     0.9113026
#   10  0.922401     0.9083892
#    5  0.9304477    0.9033943
#  1       1      0.8877167

# cv accuracy 
d2 = NULL
yourData<-train7[sample(nrow(train7)),]
folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
for (i in seq(1,10,1)) {
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_7 <- yourData[testIndexes, ]
  train_7 <- yourData[-testIndexes, ]
  testdata_7<-test_7 
  test_7[ ,21]<-NULL
  knn.model=knn(train=train_7[,1:20], test=testdata_7[,1:20], cl=train_7$solubility, k =100)
  d=table(testdata_7$solubility,knn.model)
  accuracy=sum(diag(d))/sum(d)
  d2 = rbind(d2, data.frame(i,accuracy))
}
average_accuracy=sum(d2[1:10,2])/10 
average_accuracy

# traning data ratio
#train accuracy
d1 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  knn.model=knn(train=train_7[,1:20], test=train_7[,1:20], cl=train_7$solubility, k =50)
  d=table(train_7$solubility,knn.model)
  accuracy=sum(diag(d))/sum(d)
  d1 = rbind(d1, data.frame(x,accuracy))
}
# cv accuracy
d2 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  d3 = NULL
  yourData<-train_7[sample(nrow(train_7)),]
  folds <- cut(seq(1,nrow(yourData)),breaks=10,labels=FALSE)
  for (i in seq(1,10,1)) {
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_7data <- yourData[testIndexes, ]
    train_7data <- yourData[-testIndexes, ]
    testdata_7data<-test_7data 
    test_7data[ ,21]<-NULL
    knn.model=knn(train=train_7data[,1:20], test=testdata_7data[,1:20], cl=train_7data$solubility, k =50)
    d=table(testdata_7data$solubility,knn.model)
    accuracy=sum(diag(d))/sum(d)
    d3 = rbind(d3, data.frame(i,accuracy))
  }
  average_accuracy=sum(d3[1:10,2])/10 
  d2 = rbind(d2, data.frame(x,average_accuracy))
}
d1$accuracy2=d2[,2]
write.csv(d1,"DT_11.csv")

# test data
d4 = NULL
for (x in seq(0.1,1,0.1)) {
  set.seed(5)
  TrainingDataIndex <- createDataPartition(train7[ ,21], p=x, list = FALSE)
  train_7 <- train7[TrainingDataIndex,]
  test_7<- train7[-TrainingDataIndex,]
  testdata<-train_7 
  testdata[ ,21]<-NULL
  knn.model=knn(train=train_7[,1:20], test=test7[,1:20], cl=train_7$solubility, k =50)
  d=table(test7$solubility,knn.model)
  accuracy=sum(diag(d))/sum(d)
  d4 = rbind(d4, data.frame(x,accuracy))
}
d10$KNN=d4[,2]
write.csv(d10,"DT_13.csv")