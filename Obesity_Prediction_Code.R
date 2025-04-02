#Importing libraries
library(caTools)
library(caret)
library(dummy)
library(tidyverse)
library(randomForest)
library(e1071)
library(mlbench)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(class)


#Importing dataset
obesity_df <- read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")
head(obesity_df)

View(obesity_df)

#------------------------EDA------------------------#
str(obesity_df)
summary(obesity_df)

#Category types in each column
num_categories <- obesity_df %>%  
  summarise_all(~ length(unique(.)))
print(num_categories)

#Unique values in categorical columns
unique_mtrans_values <- unique(obesity_df$MTRANS)
unique_mtrans_values

unique_CAEC_values <- unique(obesity_df$CAEC)
unique_CAEC_values


unique_CALC_values <- unique(obesity_df$CALC)
unique_CALC_values

#Calculating quantiles
Q1 <- quantile(obesity_df$Age, 0.25)
Q3 <- quantile(obesity_df$Age, 0.75)
IQR <- Q3 - Q1

#Identify outliers
outliers <- which(obesity_df$Age < 
                    (Q1 - 1.5 * IQR) | obesity_df$Age > (Q3 + 1.5 * IQR))
outliers

#Identify outliers through boxplots
#Boxplot for Age column
boxplot(obesity_df$Age, main = "Boxplot of Age column", ylab = "Age (Years)")

#Boxplot for Height column
boxplot(obesity_df$Height, main = "Boxplot of Height column", ylab = "Height (m)")

#Boxplot for Weight column
boxplot(obesity_df$Weight, main = "Boxplot of Weight column", ylab = "Weight (Kg)")

#Checking if the dataset is balanced
obesity_df$NObeyesdad <- as.factor(obesity_df$NObeyesdad)
table(obesity_df$NObeyesdad)

#------------------------Pre-processing------------------------#

#Checking for missing values in the dataset
colSums(is.na(obesity_df))

#Rounding Height, Weight and TUE columns to 2dp 
columns_to_round <- c("Height", "Weight", "TUE")
obesity_df[, columns_to_round] <- lapply(obesity_df[, columns_to_round], round, 2)

#Rounding Age, FCVC, NCP, CH2O and FAF columns to whole numbers
columns_to_round_whole <- c("Age", "FCVC", "NCP", "CH2O", "FAF")
obesity_df[, columns_to_round_whole] <- lapply(obesity_df[, columns_to_round_whole], round, 0)
View(obesity_df)

#Encoding categorical values with ONE-HOT encoding: MTRANS 
dummy_model <- dummyVars(~ MTRANS, data = obesity_df) 
encoded_data <- predict(dummy_model, newdata = obesity_df) 
obesity_df <- cbind(obesity_df, encoded_data) 
obesity_df <- obesity_df[, -which(names(obesity_df) == "MTRANS")] 
print(obesity_df)

#Encoding categorical values with DUMMY encoding: Gender, family_history_with_overweight, FAVC, SMOKE, SCC
#Selecting columns to be encoded
columns_to_encode <- c("Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC")

#Creating design matrix with dummy encoding
obesity_df_design <- model.matrix(~ . - 1, data = obesity_df[, columns_to_encode])

#Combining the design matrix with the original dataset
obesity_df <- cbind(obesity_df, obesity_df_design)

#Dropping the original columns
obesity_df <- obesity_df[, !(names(obesity_df) %in% columns_to_encode)]

#Dropping male column to prevent multicolinearity
obesity_df <- select(obesity_df, -GenderMale)

#Checking
colnames(obesity_df)

#Encoding categorical values with LABEL encoding: CAEC, CALC

#Define the mapping of levels to integers
level_mapping <- c("no" = 0, "Sometimes" = 1, "Frequently" = 2, "Always" = 3)

#Applying label encoding to 'CAEC' column
obesity_df$CAEC <- as.integer(factor(obesity_df$CAEC, levels = names(level_mapping), 
                                     labels = level_mapping))

#Applying label encoding to 'CALC' column
obesity_df$CALC <- as.integer(factor(obesity_df$CALC, levels = names(level_mapping), 
                                     labels = level_mapping))

#Checking
unique_CAEC_values <- unique(obesity_df$CAEC)
unique_CAEC_values

unique_CALC_values <- unique(obesity_df$CALC)
unique_CALC_values

colnames(obesity_df)

#------------------------Splitting dataset------------------------#

set.seed(123) #for reproducibility

#Splitting the data into train, validation, and test sets (70:15:15)

#First splitting into 70:30
sample <- sample.split(obesity_df$NObeyesdad, SplitRatio = 0.7)
train <- subset(obesity_df, sample == TRUE)
temp <- subset(obesity_df, sample == FALSE)

#Further splitting the remaining data into validation and test sets (50:50)
sample_val_test <- sample.split(temp$NObeyesdad, SplitRatio = 0.5)
validation <- subset(temp, sample_val_test == TRUE)
test <- subset(temp, sample_val_test == FALSE)

#Dimensions of the subsets
dim(train)
dim(test)
dim(validation)

obesity_train <- train
obesity_val <- validation
obesity_test <- test

#Separating features from target X and Y

#Training set
obesity_X_train <- select(obesity_train, -NObeyesdad)
obesity_Y_train <- obesity_train$NObeyesdad

#Validation set
obesity_X_val <- select(obesity_val, -NObeyesdad)
obesity_Y_val <- obesity_val$NObeyesdad

#Test set
obesity_X_test <- select(obesity_test, -NObeyesdad)
obesity_Y_test <-obesity_test$NObeyesdad

#------------------------Building Models--------------------------------#

#------------------------Decision Tree----------------------------------#
#Building the Decision Tree classifier model with the training set
decision_tree <- rpart(obesity_Y_train~., data = obesity_X_train, method = "class")

#Plotting the Decision Tree 
plot(decision_tree)
text(decision_tree)

#Using fancyRpartPlot to visually improve the Decision Tree
fancyRpartPlot(decision_tree, main="A Decision Tree classifier predicting the obesity levels of individuals", palettes="Blues", tweak=1)

#Using the Decision Tree model to make a prediction on the training set 
dt_prediction <- predict(decision_tree, obesity_X_train, type = "class")

#Returning a confusion matrix of the Decision Tree prediction results on the training set
confusionMatrix(dt_prediction, as.factor(obesity_Y_train)) 

#Using the Decision Tree model to make a prediction on the validation set 
dt_validation_prediction <- predict(decision_tree, obesity_X_val, type = "class") 

#Returning a confusion matrix of the Decision Tree prediction results on the validation set
confusionMatrix(dt_validation_prediction, as.factor(obesity_Y_val))

#------------------------Decision Tree Improvements------------------------#
#------------------------Improvement 1 - Feature Selection----------------#
#Assigning the correlations of the attributes in the validation set to a variable
correlationMatrix <- cor(obesity_X_val)

#Printing the correlationMatrix variable
print(correlationMatrix)

#Assigning the correlations of the attributes that are past the 0.5 threshold to a variable
highCorrelation <- findCorrelation(correlationMatrix, cutoff = 0.5, names = TRUE)

#Printing the highCorrelation variable
print(highCorrelation)

#Removing the attributes with a high correlation from the validation set
obesity_X_val <- select(obesity_X_val, -MTRANSPublic_Transportation)
view(obesity_X_val)

#Removing the attributes with a high correlation from the training set
obesity_X_train <- select(obesity_X_train, -MTRANSPublic_Transportation)
view(obesity_X_train)

#------------------------Improvement 2 - Tree Splitting----------------#
#Splitting the Decision Tree classifier using 'gini'
gini_decision_tree_split =  rpart(obesity_Y_train~., data = obesity_X_train, method = "class",  parms = list(split = "gini"))

#Making a prediction on the train set using the gini split Decision Tree
gini_split_train_prediction <- predict(gini_decision_tree_split, obesity_X_train, type = "class")

#Returning the confusion matrix for the gini split Decision Tree
confusionMatrix(gini_split_train_prediction, as.factor(obesity_Y_train)) 

#Making a prediction on the validation set using the gini split Decision Tree
gini_split_validation_prediction <- predict(gini_decision_tree_split, obesity_X_val, type = "class")  

#Returning the confusion matrix for the gini split Decision Tree
confusionMatrix(gini_split_validation_prediction, as.factor(obesity_Y_val)) 

#Splitting the Decision Tree classifier using 'information'
information_decision_tree_split =  rpart(obesity_Y_train~., data = obesity_X_train, method = "class",
                                         parms = list(split = "information"))

#Making a prediction on the train set using the information split Decision Tree
information_split_train_prediction <- predict(information_decision_tree_split, obesity_X_train, type = "class")

#Returning the confusion matrix for the information split Decision Tree
confusionMatrix(information_split_train_prediction, as.factor(obesity_Y_train)) 

#Making a prediction on the validation set using the information split Decision Tree
information_split_validation_prediction <- predict(information_decision_tree_split, obesity_X_val, type = "class") 

#Returning the confusion matrix for the information split Decision Tree
confusionMatrix(information_split_validation_prediction, as.factor(obesity_Y_val)) 

#------------------------Improvement 3 - Tree Pruning----------------#
#Printing the Complexity Parameter (CP) for the information split Decision Tree to find an optimal value
printcp(information_decision_tree_split) 
#Plotting the Complexity Parameter (CP)
plotcp(information_decision_tree_split) 

#Assigning the minimum cross validated error from the CP table to a variable
index <- which.min(information_decision_tree_split$cptable[, "xerror"])

#Assigning the corresponding CP value for the index variable to another variable
cp_optimal <- information_decision_tree_split$cptable[index, "CP"]

#Cross validation using rpart and 100 folds
cross_validation_rpart <- rpart(obesity_Y_train~., data=obesity_X_train, method="class", control = rpart.control(cp = cp_optimal, xval=100))

#Decision Tree pruning 
pruned_decision_tree <- prune(tree = cross_validation_rpart, cp = cp_optimal)

#Using fancyRpartPlot to visually improve the Decision Tree
fancyRpartPlot(pruned_decision_tree, main="Pruned Decision Tree classifier predicting the obesity levels of individuals", palettes="Blues", tweak=1.5)

#Making a prediction on the train set using the pruned Decision Tree
pruned_train_prediction <- predict(pruned_decision_tree, obesity_X_train, type = "class")

#Returning the confusion matrix for the pruned Decision Tree
confusionMatrix(pruned_train_prediction, as.factor(obesity_Y_train)) 

#Making a prediction on the validation set using the pruned Decision Tree
pruned_validation_prediction <- predict(pruned_decision_tree, obesity_X_val, type = "class") 

#Returning the confusion matrix for the pruned Decision Tree
confusionMatrix(pruned_validation_prediction, as.factor(obesity_Y_val)) 

#Final prediction on the validation set using the pruned Decision Tree
validation_prediction <- predict(pruned_decision_tree, obesity_X_val, type = "class")

#Convert variable to numeric type
obesity_Y_val_numeric <- as.numeric(obesity_Y_val)

#Convert variable to factor type 
obesity_Y_val_numeric <- as.factor(obesity_Y_val_numeric)

#Returning the confusion matrix for the pruned Decision Tree
confusionMatrix(validation_prediction, obesity_Y_val) 

#------------------------Decision Tree Model Testing----------------#
#Prediction on the test set using the pruned Decision Tree
testing_prediction <- predict(pruned_decision_tree, obesity_X_test, type = "class")

#Convert variable to numeric type
obesity_Y_test_numeric <- as.numeric(obesity_Y_test)

#Convert variable to factor type 
obesity_Y_test_numeric <- as.factor(obesity_Y_test_numeric)

#Returning the confusion matrix for the pruned Decision Tree
confusionMatrix(testing_prediction, obesity_Y_test) 

#------------------------Random Forest Model--------------------------------#

#Building the Random Forest model - Base model
rf_initial_model <- randomForest(obesity_Y_train~., data=obesity_X_train, proximity=TRUE) 
print(rf_initial_model)

#Predicting on train
p1 <- predict(rf_initial_model, obesity_X_train)
confusionMatrix(p1, obesity_Y_train)

#Predicting on validation
p2 <- predict(rf_initial_model, obesity_X_val)
confusionMatrix(p2, obesity_Y_val)

confusionMatrix <- table(p2, obesity_Y_val)
print(confusionMatrix)

#Precision
precision <- confusionMatrix[2,2] / 
  confusionMatrix[2,2] + 
  confusionMatrix[1,2]

print(paste("Precision: ", round(precision,2)))

#------------------------Tuning the RF model--------------------------------#
#------------------------Improvement 1 - Feature Selection------------------#
#Feature selection -  Ranking features by their importance
#Defining the control with 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(NObeyesdad~., data=obesity_train, method="lvq", preProcess="scale", trControl=control)
#Estimate variable importance
importance <- varImp(model, scale=FALSE)
#Summarise the importance level
print(importance)
#Plotting the importance
plot(importance, main="Feature Importance by Class", ylab="Features")

#Removing features with the least importance from training set
obesity_X_train = subset(obesity_X_train, select = -c(MTRANSMotorbike, MTRANSBike))

obesity_train = subset(obesity_train, select = -c(MTRANSMotorbike, MTRANSBike))

#Removing features with the least importance from validation set
obesity_X_val = subset(obesity_X_val, select = -c(MTRANSMotorbike, MTRANSBike))

obesity_val = subset(obesity_val, select = -c(MTRANSMotorbike, MTRANSBike))

#Removing features with the least importance from test set
obesity_X_test = subset(obesity_X_test, select = -c(MTRANSMotorbike, MTRANSBike))

obesity_test = subset(obesity_test, select = -c(MTRANSMotorbike, MTRANSBike))

#Building random forest model with results from feature importance appplied
rf_fs_model <- randomForest(obesity_Y_train~., data=obesity_X_train, proximity=TRUE) 
print(rf_fs_model)

#Predicting on train
p3 <- predict(rf_fs_model, obesity_X_train)
confusionMatrix(p3, obesity_Y_train)

#Predicting on validation
p4 <- predict(rf_fs_model, obesity_X_val)
confusionMatrix(p4, obesity_Y_val)

#------------------------Improvement 2 & 3 - Hyperparameter tuning------------------#
#Tuning via gridsearch for 'mtry' parameter
metric <- "Accuracy"
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(1:15))

rf_gridsearch <- train(NObeyesdad~., data=obesity_train, method="rf", 
                       metric=metric, tuneGrid=tunegrid, trControl=ctrl)
print(rf_gridsearch)
plot(rf_gridsearch)

#Predicting on train
p5 <- predict(rf_gridsearch, obesity_X_train)
confusionMatrix(p5, obesity_Y_train)

#Predicting on validation
p6 <- predict(rf_gridsearch, obesity_X_val)
confusionMatrix(p6, obesity_Y_val)

#Tuning via gridsearch for 'ntree' parameter
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=11)
modellist <- list()
for (ntree in c(1000, 1500, 2000, 2500)) {
  set.seed(123)
  fit <- train(NObeyesdad~., data=obesity_train, method="rf", 
               metric=metric, tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}

#Comparing results
results <- resamples(modellist)
summary(results)
dotplot(results)
#2000

print(modellist)

#Predicting on training 
p7 <- predict(modellist[["2000"]], newdata = obesity_X_train)
confusionMatrix(p7, obesity_Y_train)

#Predicting on validation 
p8 <- predict(modellist[["2000"]], newdata = obesity_X_val)
confusionMatrix(p8, obesity_Y_val)

#--------------------------------Final RF model------------------------------#
rf_best_model <- randomForest(x = obesity_X_train, y = obesity_Y_train, 
                              mtry = 11,
                              ntree = 2000)
print(rf_best_model)

#Predicting on validation
p9 <- predict(rf_best_model, obesity_X_val)
confusionMatrix(p9, obesity_Y_val)

#Predicting on test
p10 <- predict(rf_best_model, obesity_X_test)
confusionMatrix(p10, obesity_Y_test)

#------------------------------------The Initial KNN Model--------------------------------#
# Define the number of neighbors (k) for the KNN model
k_value <- 5  # You can choose an appropriate value based on your data and cross-validation

# Train the KNN model on the training set
knn_model <- knn(train = as.matrix(obesity_X_train), 
                 test = as.matrix(obesity_X_val), 
                 cl = obesity_Y_train, 
                 k = k_value)

# Predict on the validation set
predictions <- as.factor(knn_model)

# Evaluate on the validation set
val_predictions <- as.factor(knn(train = as.matrix(obesity_X_train), 
                                 test = as.matrix(obesity_X_val), 
                                 cl = obesity_Y_train, 
                                 k = k_value))
val_conf_matrix <- confusionMatrix(val_predictions, obesity_Y_val)
print("Validation Set:")
print(val_conf_matrix)

#------------------------------------Iteration 1 - Drop the columns-----------------------------------------#
# Define the columns to be dropped
colnames(obesity_df)
columns_to_drop <- c("MTRANSMotorbike", "MTRANSBike")

#Drop the columns from the training set
obesity_X_train <- subset(obesity_train, select = setdiff(names(obesity_X_train), columns_to_drop))

#Drop the columns from the validation set
obesity_X_val <- subset(obesity_val, select = setdiff(names(obesity_X_val), columns_to_drop))

#Drop the columns from the test set
obesity_X_test <- subset(obesity_test, select = setdiff(names(obesity_X_test), columns_to_drop))

# Define the number of neighbors (k) for the KNN model
k_value <- 5  # You can choose an appropriate value based on your data and cross-validation

# Train the KNN model on the training set
knn_model <- knn(train = as.matrix(obesity_X_train), 
                 test = as.matrix(obesity_X_val), 
                 cl = obesity_Y_train, 
                 k = k_value)

# Predict on the validation set
predictions <- as.factor(knn_model)

# Evaluate on the validation set
val_predictions <- as.factor(knn(train = as.matrix(obesity_X_train), 
                                 test = as.matrix(obesity_X_val), 
                                 cl = obesity_Y_train, 
                                 k = k_value))
val_conf_matrix <- confusionMatrix(val_predictions, obesity_Y_val)
print("Validation Set:")
print(val_conf_matrix)
#------------------------------Iteration 2 - The Scaled Model-------------------------------------#
#Standardize the features
obesity_X_train_scaled <- scale(obesity_X_train)
obesity_X_val_scaled <- scale(obesity_X_val)
obesity_X_test_scaled <- scale(obesity_X_test)

#Train the KNN model on the scaled data
knn_model_scaled <- knn(train = as.matrix(obesity_X_train_scaled), 
                        test = as.matrix(obesity_X_val_scaled), 
                        cl = obesity_Y_train, 
                        k = k_value)

# Predict on the scaled validation set
predictions_scaled <- as.factor(knn_model_scaled)

# Evaluate the model on the scaled validation set
conf_matrix_scaled <- confusionMatrix(predictions_scaled, obesity_Y_val)
print("Performance on Scaled Validation Set:")
print(conf_matrix_scaled)

#------------------------------Iteration 3 - The Tuned Model---------------------------------------#
#Create a grid of k values to search
k_grid <- data.frame(k = seq(1, 20, by = 1))

# Set up the control parameters for the grid search
ctrl <- trainControl(method = "cv", number = 5)

# Perform grid search with cross-validation
knn_tuned <- train(x = as.matrix(obesity_X_train_scaled), 
                   y = obesity_Y_train, 
                   method = "knn", 
                   tuneGrid = k_grid, 
                   trControl = ctrl)

#Print the tuned parameters and performance
print(knn_tuned)

#Get predictions on the validation set
predictions_val <- predict(knn_tuned, newdata = as.matrix(obesity_X_val_scaled))

#Confusion matrix and performance metrics
conf_matrix <- confusionMatrix(predictions_val, obesity_Y_val)

#Print the confusion matrix and metrics
print(conf_matrix)


#Get predictions on the Test set
predictions_test <- predict(knn_tuned, newdata = as.matrix(obesity_X_test_scaled))

#Confusion matrix and performance metrics
conf_matrix <- confusionMatrix(predictions_test, obesity_Y_test)

#Print the confusion matrix and metrics
print(conf_matrix)
#------------------------------------------------------------------------------#