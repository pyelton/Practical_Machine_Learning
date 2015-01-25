library(caret)
setwd("~/Dropbox/PEPPERCOPY/Practical_machine_learning")

#Load data files
train <- read.csv("pml-training.csv", header = TRUE, row.names = 1, colClasses = c(rep("factor",2), rep("numeric",2), rep("factor", 2), rep("character", 153)))
test <- read.csv("pml-testing.csv", header = TRUE, row.names = 1, colClasses = c(rep("factor",2), rep("numeric",2), rep("factor", 2), rep("character", 153)))
modFit = train(classe ~ ., method="rpart", data=train) 

#Data preprocessing/clean up
#Replace empty values with NA
train[train == ""] <- NA
test[test == ""] <- NA

#Remove columns containing NAs by looking through the data frame by column
training = train[ , ! apply(train , MARGIN = 2 , function(x) any(is.na(x)) ) ]
testing = test[, ! apply(test, MARGIN = 2, function(x) any(is.na(x)))]

#Remove timestamps, window numbers, and name of subject 
#Setting the columns to numeric values
t = as.data.frame(sapply(training[7:58], as.numeric))
final = cbind(training[59], t)
tes = as.data.frame(sapply(testing[7:58], as.numeric))
ftest = cbind(testing[59], tes)

#Split the data into training and testing sets within the training set
set.seed(333)
inTrain <- createDataPartition(y=final$classe, p=0.6, list=FALSE)
trainFit <- final[inTrain,]
testFit <- final[-inTrain,]

#Examine other variables for relationship to classe
featurePlot(x=trainFit[,c("roll_belt","pitch_belt","yaw_belt", "total_accel_belt", 
            "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x",
            "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y")],
            y = trainFit$classe, plot="box")
#both yaw_belt and roll_belt are higher for incorrect classes
#magnet_belt_y is lower for E
featurePlot(x=trainFit[,c("magnet_belt_z",  "roll_arm",  
            "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", 
            "gyros_arm_y",  "gyros_arm_z")], 
            y = trainFit$classe, plot="box")
#magnet_belt_z is higher for E
featurePlot(x=trainFit[,c("accel_arm_x", "accel_arm_y", "accel_arm_z", 
            "magnet_arm_x", "magnet_arm_y", "magnet_arm_z",        
            "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell")], 
            y = trainFit$classe, plot="box")
#roll_dumbbell is higher for B and E 
featurePlot(x=trainFit[,c("total_accel_dumbbell", "gyros_dumbbell_x", "gyros_dumbbell_y",    
                    "gyros_dumbbell_z", "accel_dumbbell_x", "accel_dumbbell_y",    
                    "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y"   )], 
            y = trainFit$classe, plot="box")
#magnet_dumbbell_x and magnet_dumbbell_y is higher for B
featurePlot(x=trainFit[,c("magnet_dumbbell_z","roll_forearm","pitch_forearm",       
            "yaw_forearm","total_accel_forearm","gyros_forearm_x")], 
            y = trainFit$classe, plot="box")
#yaw_forearm higher for C
featurePlot(x=trainFit[,c("gyros_forearm_y","gyros_forearm_z","accel_forearm_x",     
            "accel_forearm_y","accel_forearm_z","magnet_forearm_x",    
            "magnet_forearm_y","magnet_forearm_z"  )], 
            y = trainFit$classe, plot="box")
#magnet_forearm_x is lower for D and E

#Look for correlations among predictor variables
library(Hmisc)
correlations <- rcorr(as.matrix(trainFit[2:53]), type = "spearman")
correlations$r[correlations$r > 0.8 & correlations$r  < 1.0]
#A small subset of 8 variables are highly correlated with one another > 0.8 spearman correlation

#So we try PCA to combine these variables 
preProc <- preProcess(trainFit[,-1], method = "pca", pcaComp=2) #pcaComp number of PCs to compute
fitPC <- predict(preProc, trainFit[,-1])

#Predict principal components for training set
qplot(fitPC[,1],fitPC[,2], col = trainFit$classe)
modfitPC <- train(trainFit$classe ~ . , data = fitPC, method = "gbm", trControl = trainControl(method = "cv", number = 10))
testPC <- predict(preProc, testFit[,-1])
PCpred <- predict(modfitPC, testPC)
confusionMatrix(testFit$classe, predict(modfitPC, testPC))
plot(PCpred ~ testFit$classe)


#both yaw_belt and roll_belt are higher for incorrect classes
#magnet_belt_y is lower for E
#magnet_belt_z is higher for E
#roll_dumbbell is higher for B and E 
#magnet_dumbbell_x and magnet_dumbbell_y is higher for B
#yaw_forearm higher for C
#magnet_forearm_x is lower for D and E

#Fit a model with just these predictor variables
modfitsubset <- train(classe ~ yaw_belt + roll_belt + magnet_belt_y + magnet_belt_z + roll_dumbbell + magnet_dumbbell_x + magnet_dumbbell_y + yaw_forearm + magnet_forearm_x, data = trainFit, method = "gbm", trControl = trainControl(method = "cv", number = 3))
confusionMatrix(testFit$classe, predict(modfitsubset, testFit))

#Fit a model with all of the possible explanatory variables
modfit <- train(classe ~ ., data = trainFit, method = "gbm", trControl = trainControl(method = "cv", number = 3))
testall <- predict(modfit, testFit)
confusionMatrix(testFit$classe, predict(modfit, testFit))
#96% out of sample accuracy
plot(testall ~ testFit$classe)


#Test model on 20 test cases and submit predictions
testanswers <- as.vector(predict(modfit, ftest))
answers = testanswers

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(answers)
