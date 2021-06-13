# Housing-Price-Prediction
Machine Learning (ML) model for price prediction using Linear Regression

## Description

This code was written in MATLAB for the [competition presented by Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview).
The proposed ML model was developed in order to represent one of the possible solutions for the housing price prediction problem. 

## Dataset

The dataset provided by Kaggle consists of 2919 samples with 79 features each. This dataset originally split into training and testing datasets with 1460 and 1459 samples, respectively. In order to justify our models performance, the training dataset is split into two subsets of data. One subset contains 86% of the original training data and is used to train our model, second subset that is called validation subset contains the remaining 14% and is used to validate our model. The accuracy of validation with the 14% of the training data will provide us with an understanding of the efficiency of our design.

## Data preprocessing

Data preprocessing consists of the following steps:
* The data is cleaned from features that contained more than 50% of missing data;
* All categorical features are transformed into numerical features;
* The features are sorted so it would be possible to describe them linearly;
* Some features with very low variance are deleted;
* Outliers are deleted;
* All missing values are found and changed to either 0 or most frequent values of the features that contain these missing values, wherever it makes sence.
