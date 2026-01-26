# Author: Marshall Smith

# Description: 
# Comprehensive modeling project in R, demonstrating regression,
# ensemble, classification, and deep learning techniques.

# Includes:
#  - PCR, PLSR, Lasso Regression
#  - Bagging and Random Forest
#  - Support Vector Machines (Linear, Radial, Polynomial)
#  - Recurrent Neural Network (RNN) on NYSE Data
#  - Convolution Neural Network (CNN) for Image Classification


#########################################
# SECTION 1: ENVIRONMENT SETUP AND DATA IMPORT
#########################################

rm(list = ls())

setwd("~/Desktop/School/R_GitHub")

# check environment
getwd()
list.files()


#########################################
# SECTION 2: DATA PREPERATION FOR BODYDF
#########################################

# Read txt file and convert to data frame
BodyDF <- data.frame(read.table("body.dat.txt", header = FALSE))

colnames(BodyDF) <- c(
  "BiacromialDiam", "BiiliacDiam", "BitrochantericDiam", "ChestDepth", 
  "ChestDiam", "ElbowDiam", "WristDiam", "KneeDiam", "AnkleDiam",
  "ShoulderGirth", "ChestGirth", "WaistGirth", "NavelGirth", "HipGirth",
  "ThighGirth", "BicepGirth", "ForearmGirth", "KneeGirth",
  "CalfMaxGirth", "AnkleMinGirth", "WristMinGirth",
  "Age", "Weight", "Height", "Gender"
)

# Define predictors (X) and responses (Y)
Y <- cbind(BodyDF$Age, BodyDF$Weight, BodyDF$Height, BodyDF$Gender)
X <- cbind(
  BodyDF$BiacromialDiam, BodyDF$BiiliacDiam, BodyDF$BitrochantericDiam,
  BodyDF$ChestDepth, BodyDF$ChestDiam, BodyDF$ElbowDiam, BodyDF$WristDiam,
  BodyDF$KneeDiam, BodyDF$AnkleDiam, BodyDF$ShoulderGirth, BodyDF$ChestGirth,
  BodyDF$WaistGirth, BodyDF$NavelGirth, BodyDF$HipGirth, BodyDF$ThighGirth,
  BodyDF$BicepGirth, BodyDF$ForearmGirth, BodyDF$KneeGirth,
  BodyDF$CalfMaxGirth, BodyDF$AnkleMinGirth, BodyDF$WristMinGirth
)

head(BodyDF)


#########################################
# SECTION 3: EXPLORATORY VISUALIZATION
#########################################

# Create scatter plot to visualize gender differences
# combining response variables into dataframe
Y_df <- data.frame(
  Age = Y[,1],
  Weight = Y[,2],
  Height = Y[,3],
  Gender = as.factor(Y[,4])
)

# Scatter plot
plot(
  Y_df$Height, Y_df$Weight, col = Y_df$Gender,
  pch = 19, xlab = "Height (cm)", ylab = "Weight (kg)",
  main = "Height vs Weight Colored by Gender"
)

legend("topleft", legend = levels(Y_df$Gender),
       col = 1:length(levels(Y_df$Gender)), pch = 19)

# By plotting weight against height we uncover 2 distinct clusters
# - Taller/heavier group likely represents males (coded as 1)
# - Shorter/lighter group likely represents females (coded as 2)


#########################################
# SECTION 4: REGRESSION MODELING (PCR, PLSR, LASSO)
#########################################

library(pls)  # For pcr and plsr
set.seed(42) # Ensures reproducibility

# Define response variable (Weight) and predictor matrix
Y <- BodyDF$Weight

X <- cbind(
  BodyDF$BiacromialDiam, BodyDF$BiiliacDiam, BodyDF$BitrochantericDiam,
  BodyDF$ChestDepth, BodyDF$ChestDiam, BodyDF$ElbowDiam, BodyDF$WristDiam,
  BodyDF$KneeDiam, BodyDF$AnkleDiam, BodyDF$ShoulderGirth, BodyDF$ChestGirth,
  BodyDF$WaistGirth, BodyDF$NavelGirth, BodyDF$HipGirth, BodyDF$ThighGirth,
  BodyDF$BicepGirth, BodyDF$ForearmGirth, BodyDF$KneeGirth,
  BodyDF$CalfMaxGirth, BodyDF$AnkleMinGirth, BodyDF$WristMinGirth
)

# split data into training and testing sets
# test will have 200 observations while train will have the remaining 307

# total rows
n <- nrow(BodyDF)
# Random sample of 200 indices for test
test_idx <- sample(1:n, 200)

# Training and test sets
X_train <- X[-test_idx, ]
Y_train <- Y[-test_idx]

X_test <- X[test_idx, ]
Y_test <- Y[test_idx]


#########################################
# 4.1 Principal Components Regression (PCR)
#########################################

# Fit PCR Model
pcr_model <- pcr(Y_train ~ X_train, scale = TRUE, validation = "CV")
summary(pcr_model)

# Plot RMSE to choose component
validationplot(pcr_model, val.type = "RMSEP", main = "PCR - RMSE by Components")

# PCR Prediction (ncomp = 10)
pcr_pred <- predict(pcr_model, newdata = X_test, ncomp = 10)
# PCR MSE
pcr_mse <- mean((pcr_pred - Y_test)^2)


#########################################
# 4.2 Partial Least Squares Regression
#########################################

# fit PLSR model
plsr_model <- plsr(Y_train ~ X_train, scale = TRUE, validation = "CV")
summary(plsr_model)

# Plot RMSE to choose components
validationplot(plsr_model, val.type = "RMSEP", main = "PLSR - RMSE by Components")

# PLSR Prediction (ncomp = 10) on test set
plsr_pred <- predict(plsr_model, newdata = X_test, ncomp = 10)
# PLSR MSE
plsr_mse <- mean((plsr_pred - Y_test)^2)



#########################################
# Findings and Explanation
#########################################

# After running summary() on both the pcr and plsr models, we compared the percentage
# of variance explained in the response variable (Weight) across the number of components. 
# We observed that the PLSR model consistently explained more of the y variance than the 
# PCR model, especially in the first several components. While both models eventually
# explained around 96% of the variance, PLSR achieved this with fewer components. 
# This pattern is not surprising. PCR creates components that explain the most variance
# in the predictor variables (X) without considering their relevance
# to the response (Y). As a result, PLSR explicitly builds components to maximize
# covariance between X and Y, making it more efficient at capturing predictive structure
# early on. 

# Based on the cross-validated RMSEP and the percentage of variance explained in
# the response variable, I selected:
#   - 6 components for the PCR model, since it achieved a good trade-off between
#   prediction accuracy (adjCV RMSEP = 2.988) and simplicity, with 95.11% of Y variance explained.
#   - 5 components for the PLSR model, as it yielded the lowest cross-validated RMSEP (2.852)
#   and already captured 95.97% of the Y variance


#########################################
# 4.3 Lasso Regression for Variable Selection
#########################################

library(glmnet)

# Convert X_train to matrix (required by glmnet)
x_train_mat <- as.matrix(X_train)
y_train_vec <- Y_train  # already a vector

# Fit lasso model with cross-validation
set.seed(42)
lasso_cv <- cv.glmnet(x_train_mat, y_train_vec, alpha = 1, standardize = TRUE)

# Plot cross-validation results
plot(lasso_cv)

# Optimal lambda (minimizing CV error)
best_lambda <- lasso_cv$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Coefficients at optimal lambda
lasso_coef <- coef(lasso_cv, s = best_lambda)
print(lasso_coef)

# Predict on X_test using optimal lambda
x_test_mat <- as.matrix(X_test)
lasso_preds <- predict(lasso_cv, newx = x_test_mat, s = best_lambda)

# Calculate RMSE and MSE
lasso_rmse <- sqrt(mean((Y_test - lasso_preds)^2))
lasso_mse <- mean((lasso_preds - Y_test)^2)

cat("Lasso RMSE on test set:", lasso_rmse, "\n")
cat("Lasso MSE on test set:", lasso_mse, "\n")


#########################################
# 4.4 Model Comparison (PCR vs PLSR vs Lasso)
#########################################

pcr_ncomp = 6
plsr_ncomp = 5

# recalculating RMSE for consistency
# Predict on test set
pcr_preds <- predict(pcr_model, newdata = X_test, ncomp = pcr_ncomp)
plsr_preds <- predict(plsr_model, newdata = X_test, ncomp = plsr_ncomp)


# Calculate RMSE
pcr_rmse <- sqrt(mean((Y_test - pcr_preds)^2))
plsr_rmse <- sqrt(mean((Y_test - plsr_preds)^2))

# Final Comparison Table
rmse_results <- data.frame(
  Method = c("PCR", "PLSR", "Lasso"),
  RMSE = c(pcr_rmse, plsr_rmse, lasso_rmse)
)
print(rmse_results)

# Based on the RMSE, the best method was PLSR which had an rmse = 2.699730. 
# The second best method in predction is the Lasso regression model which had 
# RMSE = 2.720155. This means that PLSR did the best job at predicting weight. 


#########################################
# SECTION 5: ENSEMBLE MODELING (BAGGING & RANDOM FOREST)
#########################################

library(randomForest)
set.seed(42)

# Prepare training and testing data
train_data <- data.frame(X_train, weight = Y_train)
test_data <- data.frame(X_test, weight = Y_test)
colnames(X_test) <- colnames(X_train)

# Number of predictors
p <- ncol(X_train)


#########################################
# 5.1: Bagging Model
#########################################

X_test <- as.data.frame(X_test)


# fit bagging
bagging_model <- randomForest(
  weight ~ ., 
  data = train_data,
  mtry = p,
  ntree = 500,
  keep.inbag = TRUE,
  keep.forest = TRUE,
  xtest = X_test,
  ytest = Y_test
)


#########################################
# 5.2: Random Forest Model
#########################################

# fit random forest
rf_model <- randomForest(
  weight ~ ., 
  data = train_data,
  mtry = floor(sqrt(p)),  # Default for RF
  ntree = 500,
  keep.inbag = TRUE,
  keep.forest = TRUE,
  xtest = X_test,
  ytest = Y_test
)


#########################################
# 5.3: Model Performance Visualization
#########################################

# Extract MSE for each model
bagging_mse <- bagging_model$test$mse
rf_mse <- rf_model$test$mse

# Plot comparison
plot(bagging_mse, type = "l", col = "blue", lwd = 2,
     ylab = "Test MSE", xlab = "Number of Trees",
     main = "Test MSE vs Number of Trees")
lines(rf_mse, col = "red", lwd = 2)
legend("topright", legend = c("Bagging", "Random Forest"),
       col = c("blue", "red"), lwd = 2)

# Interpretation:
# The MSE stabilizes as tree count increases, with Random Forest
# typically achieving slightly better generalization performance.


#########################################
# 5.4: Variable Importance
#########################################

# Random Forest importance
importance(rf_model)
varImpPlot(rf_model, main = "Variable Importance - Random Forest")

# Bagging importance
importance(bagging_model)
varImpPlot(bagging_model, main = "Variable Importance - Bagging")

# Get variable importance from both models
rf_importance <- importance(rf_model)
bagging_importance <- importance(bagging_model)

# Convert to data frames
rf_df <- data.frame(Variable = rownames(rf_importance), RF_Importance = rf_importance[, 1])
bagging_df <- data.frame(Variable = rownames(bagging_importance), Bagging_Importance = bagging_importance[, 1])

# Sort and get top 3 for each
top_rf <- rf_df[order(-rf_df$RF_Importance), ][1:3, ]
top_bagging <- bagging_df[order(-bagging_df$Bagging_Importance), ][1:3, ]

# Create combined table
top_table <- data.frame(
  Rank = 1:3,
  RF_Variable = top_rf$Variable,
  RF_Importance = round(top_rf$RF_Importance, 2),
  Bagging_Variable = top_bagging$Variable,
  Bagging_Importance = round(top_bagging$Bagging_Importance, 2)
)

# Print the table
print(top_table)

# Findings:
# Both models identify X12, X11, and X16 as key predictors for weight.
# Bagging distributes importance more evenly, while Random Forest
# highlights fewer dominant predictors due to variable sampling.


##########################################
# 5.5: Model Comparison with Regression Models
##########################################

# Calculate RF MSE
rf_final_mse <- rf_model$test$mse[500]

# Retrieve previous MSEs
lasso_mse <- lasso_rmse^2
pcr_mse <- pcr_rmse^2
plsr_mse <- plsr_rmse^2

mse_table <- data.frame(
  Model = c("Lasso", "PCR", "PLSR", "Random Forest"),
  MSE = round(c(lasso_mse, pcr_mse, plsr_mse, rf_final_mse), 4)
)

print(mse_table)

# Interpretation: 
# Random Forest performs competitively but definitely underperforms
# compared to PLSR and Lasso.
# Ensemble methods improve stability and handle nonlinearities, but bias remains 
# due to variable correlations.


#########################################
# SECTION 6: CLASSIFICATION MODELING (SVM)
#########################################
rm(list = ls())

library(ISLR2)
library(e1071)
set.seed(42)


#########################################
# 6.1: Data Preparation (OJ Dataset)
#########################################

# Load OJ dataset from ISLR2 package
data(OJ)
?OJ

# Split into training (800 obs) and testing sets
train_idx <- sample(1:nrow(OJ), 800)
OJ_train <- OJ[train_idx, ]
OJ_test <- OJ[-train_idx, ]

# Ensure categorical response
OJ_train$Purchase <- as.factor(OJ_train$Purchase)


#########################################
# 6.2: Linear Support Vector Classifier
#########################################

# Fit the support vector classifier (linear kernel by default)
svm_linear <- svm(Purchase ~ ., data = OJ_train, cost = 0.01, kernel = "linear")
summary(svm_linear)

# Train and Test predictions
train_preds_linear <- predict(svm_linear, OJ_train)
test_preds_linear <- predict(svm_linear, OJ_test)

# Train and Test Error Rates
train_error_linear <- mean(train_preds_linear != OJ_train$Purchase)
test_error_linear <- mean(test_preds_linear != OJ_test$Purchase)


# Printing error rates
cat("Linear SVM Training Error:", round(train_error_linear, 4), "\n")
cat("Linear SVM Test Error:", round(test_error_linear, 4), "\n")


# Interpretation:
# - Training Error: 0.1713
# - Test Error: 0.163
# Model uses 432 support vectors and has 2 classes (CH & MM)


#########################################
# 6.3: Tuning Linear SVM (Cross-Validation)
#########################################3


set.seed(42)
tune_linear <- tune(svm, Purchase ~ ., data = OJ_train,
                    kernel = "linear",
                    ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))

# Summary
summary(tune_linear)


# Finding best linear model
svm_linear_best <- tune_linear$best.model

# Predictions
train_pred_best_linear <- predict(svm_linear_best, OJ_train)
test_pred_best_linear <- predict(svm_linear_best, OJ_test)

# Error rates
train_error_best_linear <- mean(train_pred_best_linear != OJ_train$Purchase)
test_error_best_linear <- mean(test_pred_best_linear != OJ_test$Purchase)

# Print error rates
cat("Tuned Linear SVM Training Error:", round(train_error_best_linear, 4), "\n")
cat("Tuned Linear SVM Test Error:", round(test_error_best_linear, 4), "\n")

# Tuned Linear SVM Training Error: 0.1675 
# Tuned Linear SVM Test Error: 0.163 
# So we can see the tu ing has an effect on the training error but not as much on
# the test error

#########################################
# 6.4: Radial Kernal SVM
#########################################
# use default value for gamma

# fit a radial SVM with cost = 0.01
svm_radial <- svm(Purchase ~ ., data = OJ_train, kernel = "radial", cost = 0.01)

# Summary of the model
summary(svm_radial)

# computing training and test error rates

# Train & Test predictions
train_pred_radial <- predict(svm_radial, OJ_train)
test_pred_radial <- predict(svm_radial, OJ_test)

# Train & Test Error Rate
train_error_radial <- mean(train_pred_radial != OJ_train$Purchase)
test_error_radial <- mean(test_pred_radial != OJ_test$Purchase)

# Print results
cat("Radial SVM (cost=0.01) Training Error:", round(train_error_radial, 4), "\n")
cat("Radial SVM (cost=0.01) Test Error:", round(test_error_radial, 4), "\n")

# Tuning the cost parameter
set.seed(42)
tune_radial <- tune(svm, Purchase ~ ., data = OJ_train, kernel = "radial",
                    ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))

# Best model summary
summary(tune_radial)

# Fitting best model and computing final errors
# Extract best model
svm_radial_best <- tune_radial$best.model


# Predictions
train_pred_best_radial <- predict(svm_radial_best, OJ_train)
test_pred_best_radial <- predict(svm_radial_best, OJ_test)

# Error rates
train_error_best_radial <- mean(train_pred_best_radial != OJ_train$Purchase)
test_error_best_radial <- mean(test_pred_best_radial != OJ_test$Purchase)

# printing final errors
cat("Tuned Radial SVM Training Error:", round(train_error_best_radial, 4), "\n")
cat("Tuned Radial SVM Test Error:", round(test_error_best_radial, 4), "\n")

# The radial SVM (cost) Training Error: 0.385
# The radial SVM (cost=0.01) Test Error: 0.4037

# The Tuned Radial SVM Training Error: 0.15
# The Tuned Radial SVM Test Error: 0.1593


#########################################
# 6.5: Polynomial Kernel SVM (Degree = 2)
#########################################

# fitting polynomial svm with cost = 0.01 and degree = 2
svm_poly <- svm(Purchase ~ ., data = OJ_train, kernel = "polynomial", degree = 2, cost = 0.01)

# View summary
summary(svm_poly)


# Train & Test Predictions
train_pred_poly <- predict(svm_poly, OJ_train)
test_pred_poly <- predict(svm_poly, OJ_test)

# Train and Test error rates
train_error_poly <- mean(train_pred_poly != OJ_train$Purchase)
test_error_poly <- mean(test_pred_poly != OJ_test$Purchase)

# Print error rates
cat("Polynomial SVM (cost = 0.01, degree = 2) Training Error:", round(train_error_poly, 4), "\n")
cat("Polynomial SVM (cost = 0.01, degree = 2) Test Error:", round(test_error_poly, 4), "\n")


# Tune polynomial model
set.seed(42)
tune_poly <- tune(svm, Purchase ~ ., data = OJ_train,
                  kernel = "polynomial",
                  degree = 2,
                  ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))

# Best model info
summary(tune_poly)


# Extracting best model
svm_poly_best <- tune_poly$best.model

# Predictions
train_pred_poly_best <- predict(svm_poly_best, OJ_train)
test_pred_poly_best <- predict(svm_poly_best, OJ_test)

# Error rates
train_error_poly_best <- mean(train_pred_poly_best != OJ_train$Purchase)
test_error_poly_best <- mean(test_pred_poly_best != OJ_test$Purchase)

# Print error rates
cat("Tuned Polynomial SVM Training Error:", round(train_error_poly_best, 4), "\n")
cat("Tuned Polynomial SVM Test Error:", round(test_error_poly_best, 4), "\n")

# Polynomial SVM (cost = 0.01, degree = 2) Training Error: 0.385
# Polynomial SVM (cost = 0.01, degree = 2) Test Error: 0.4037

# Tuned Polynomial SVM Training Error: 0.1475
# Tuned Polynomial SVM Test Error: 0.1667


#########################################
# 6.6: Kernel Comparison Summary
#########################################

# Create a summary table
svm_results <- data.frame(
  Kernel = c("Linear", "Radial", "Polynomial"),
  Test_Error = c(test_error_best_linear, test_error_best_radial, test_error_poly_best)
)

# Print the table
print(svm_results)

# The Radial kernel SVM gives the lowest test error rate, suggesting it performed 
# slightly better than both the Linear and Polynomial kernel SVMs on this dataset.
# This can indicate that nonlinear decision boundaries capture relationships
# between predictor variables and purchase choice effectively





#########################################
# SECTION 7: DEEP LEARNING APPLICATIONS
#########################################
# This section explores modern deep learning methods:
# (1) Recurrent Neural Networks (RNNs) for time-series data
# (2) Convolutional Neural Networks (CNNs) for image classification
#########################################


#########################################
# 7.1: Recurrent Neural Networks (RNN) - NYSE Time Series
#########################################
rm(list = ls())



#########
#########################################
# RESET RETICULATE + VIRTUALENV SETUP
# May need to load for virtual environment (just uncomment)
#########################################
#library(reticulate)

# Remove old broken env
#virtualenv_remove("r-reticulate")  # type Y

# Install Python 3.10 via reticulate (isolated)
#install_python(version = "3.10")

# Create new virtualenv with Python 3.10
#virtualenv_create("r-reticulate", python = reticulate::py_discover_config()$python)

# Activate the env
#use_virtualenv("r-reticulate", required = TRUE)

# Install working versions
#py_install(
  #packages = c(
    #"tensorflow==2.13.0",
    #"keras==2.13.1",
    #"Pillow",
    #"numpy<2"
  #),
#  envname = "r-reticulate",
#  pip = TRUE
#)

# Verify
#py_config()



#install.packages("keras")   # make sure it installs the latest from CRAN

library(reticulate)
library(keras)
library(tensorflow)

use_virtualenv("r-reticulate", required = TRUE)


keras::backend()  # should show a backend pointing to your TF 2.13 env

##########




data(NYSE)
?NYSE
# Prepare NYSE data
xdata <- data.matrix(
  NYSE[, c("DJ_return", "log_volume", "log_volatility")]
)
istrain <- NYSE[, "train"]
xdata <- scale(xdata)

# Add day_of_week as factor
NYSE$day_of_week <- factor(NYSE$day_of_week)


#########################################
# 7.1.1: Creating Lagged Time-Series Features
#########################################

# Create lagged matrix function
lagm <- function(x, k = 1) {
  n <- nrow(x)
  pad <- matrix(NA, k, ncol(x))
  rbind(pad, x[1:(n - k), ])
}


# Create autoregressive frame
arframe <- data.frame(
  log_volume = xdata[, "log_volume"],
  L1 = lagm(xdata, 1),
  L2 = lagm(xdata, 2),
  L3 = lagm(xdata, 3),
  L4 = lagm(xdata, 4),
  L5 = lagm(xdata, 5)
)

# Removing initial rows with NA from lagging
arframe <- arframe[-(1:5), ]
istrain <- istrain[-(1:5)]


#########################################
# 7.1.2: Encoding Day-of-Week Information
#########################################

# add day_of_week
arframed <- data.frame(day = NYSE[-(1:5), "day_of_week"], arframe)

# Create dummy matrix for day_of_week
day_mat <- model.matrix(~ day - 1, data = arframed)

colnames(day_mat)
colnames(arframed)
colnames(NYSE)


#########################################
# 7.1.3: Preparing RNN Input Array
#########################################

# Get x input for RNN (lagged variables only)
xrnn <- data.matrix(arframe[, -1])  # drop log_volume
n <- nrow(arframe)

# reshape into array (samples, timesteps, features)
xrnn <- array(xrnn, c(n, 3, 5))
xrnn <- xrnn[, , 5:1]
xrnn <- aperm(xrnn, c(1, 3, 2))  # shape: (n, 5 lags, 3 variables)


#########################################
# 7.1.4: RNN Model Architecture
#########################################

# Combine RNN input and day dummy input in a duel-iput model

# RNN Branch input(5 time steps, 3 variables)
input_seq <- layer_input(shape = c(5, 3))

# day_of_week input: one-hot encoded
input_day <- layer_input(shape = ncol(day_mat))

# RNN path
rnn_branch <- input_seq %>%
  layer_simple_rnn(units = 12, dropout = 0.1, recurrent_dropout = 0.1)

# Combine RNN output with day info
combined <- layer_concatenate(list(rnn_branch, input_day)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

# Define model
rnn_model <- keras_model(inputs = list(input_seq, input_day), outputs = combined)

# Compile model
rnn_model$compile(
  loss = "mse",
  optimizer = optimizer_rmsprop()
)

class(rnn_model)

#########################################
# 7.1.5: Training the RNN Model
#########################################


length(istrain)             # number of TRUE/FALSE in logical index
dim(xrnn)                   # check first dimension
dim(day_mat)                # check first dimension
length(arframe$log_volume)  # check length

# Ensure all arrays have the same rows
x_train <- list(
  xrnn[istrain, , , drop = FALSE],
  day_mat[istrain, , drop = FALSE]
)

x_val <- list(
  xrnn[!istrain, , , drop = FALSE],
  day_mat[!istrain, , drop = FALSE]
)

y_train <- array(arframe[istrain, "log_volume"], dim = c(sum(istrain), 1))
y_val   <- array(arframe[!istrain, "log_volume"], dim = c(sum(!istrain), 1))


# Fit model
history <- rnn_model$fit(
  x = x_train,
  y = y_train,
  batch_size = 64L,
  epochs = 75L,
  validation_data = list(x_val, y_val)
)


#########################################
# 7.1.6: Model Evaluation (Test R^2)
#########################################
# Predict and compute R²
kpredd <- rnn_model$predict(x_val)


# kpredd <- predict(rnn_model, list(xrnn[!istrain, , ], day_mat[!istrain, ]))
V0 <- var(arframe[!istrain, "log_volume"])
r2 <- 1 - mean((kpredd - arframe[!istrain, "log_volume"])^2) / V0
cat("Test R² with day_of_week in RNN:", r2, "\n")

# Result:
# The Test R^2 with day_of_week in RNN: 0.4503209
# Including day-of-week information improves predictive performance,
# capturing temporal patterns in trading behavior.

#########################################
# 7.2: Convolutional Neural Network (CNN) - Image Classification
#########################################

# Load required libraries
library(magick)
library(tibble)
library(magrittr)

#########################################
# 7.2.1: Load VGG16 Model
#########################################

# Load CNN Model
cnn_model <- application_vgg16(weights = "imagenet")

#########################################
# 7.2.2: Image Prediction Function
#########################################

# Prediction function
predict_top5 <- function(img_path) {
  img <- image_load(img_path, target_size = c(224, 224)) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, 224, 224, 3)) %>%
    imagenet_preprocess_input()
  
  preds <- cnn_model %>% predict(img)
  imagenet_decode_predictions(preds, top = 5)[[1]]
}

# Check to see if function is working
predict_top5(img_path = "my_animals/img9.jpg")
#########################################
# 7.2.3: Classifying Images
#########################################

# Load all .jpg or .JPG files
image_files <- list.files("my_animals", full.names = TRUE, pattern = "(?i)\\.jpg$")

# check how many were found
cat("Found", length(image_files), "image(s).\n")

# Loop through each image
for (i in seq_along(image_files) ) {
  cat("\n=========================================\n")
  cat("Top 5 predictions for:", basename(image_files[i]), "\n")
  
  result <- tryCatch({
    predict_top5(image_files[i])
  }, error = function(e) {
    message("Error processing image: ", basename(image_files[i]))
    return(NULL)
  })
  
  if (!is.null(result)) {
    print(as_tibble(result))
  }
}



#########################################
# SECTION 7 SUMMARY
#########################################
# - RNN effectively modeled temporal financial data,
# achieving a test R^2 of approximately 0.4503209.
# - CNNs demonstrated transfer learning via VGG16,
# accurately classifying real-world images without pretraining.
# - These models showcase advanced deep learning techniques
# applicable to both time-series forecasting and computer vision.




