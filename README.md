# Housing-Price-Prediction
Machine Learning (ML) model for price prediction using Linear Regression

## Description

This code was written in MATLAB for the [competition presented by Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview).
The proposed ML model was developed in order to represent one of the possible solutions for the housing price prediction problem. 

## Dataset

The dataset provided by Kaggle consists of 2919 samples with 79 features each. This dataset originally split into training and testing datasets with 1460 and 1459 samples, respectively. In order to justify our models performance, the training dataset is split into two subsets of data. One subset contains 86% of the original training data and is used to train our model, second subset that is called validation subset contains the remaining 14% and is used to validate our model. The accuracy of validation with the 14% of the training data will provide us with an understanding of the efficiency of our design.

## Data preprocessing

Feature Engineering
On the preprocessing step the data was cleaned from features that contained more than 50% of missing data; all categorical features were transformed into numerical features; after that they were sorted so it would be possible to describe them linearly; some features with low variance were also deleted; some data samples have features with values that vary widely from the average values, this data samples were also deleted; and, lastly, we found all missing values and change them either to 0 or to most frequent values of the features that contain these missing values. As a prediction model it was decided to use linear regression since the given data can be described linearly. On evaluation step, RMSLE was used and gave result of 0.1402.
