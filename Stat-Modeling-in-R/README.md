# Comprehensive Statistical Modeling and Machine Learning in R

## Overview
This project explores advanced modeling techniques in R, spanning regression, classification, and deep learning. It combines structured, temporal, and image data to demonstrate data-driven prediction and analysis.

### Datasets
- 'body.dat.txt' (anthropometric measurements)
- 'OJ' (consumer dataset from ISLR2 package)
- 'NYSE' (financial dataset from ISLR2 package)

### Methods Used
- Principal Component Regression (PCR)
- Partial Least Squares Regression (PLSR)
- Lasso Regression
- Bagging and Random Forests
- Support Vector Machines (SVM) - Linear, Radial, Polynomial
- Recurrent Neural Networks (RNN) with Keras
- COnvolution Neural Networks (CNN) for image classification

## Key Results
- **PLSR model achieved lowest RMSE (2.70)** for body weight prediction
- **Radial SVM achieved lowest test error (0.159)** for brand classification
- **RNN achieved test R^2 = 0.45**, capturing temporal financial patterns
- **CNN successfully classified animal images** using transfer learning

### Tools and Libraries
R, pls, glmnet, randomForest, e1071, keras, magick, ISLR2

## Author
Marshall Smith
