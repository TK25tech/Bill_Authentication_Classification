Banknote Authentication Using Machine Learning
Project Overview

This project was completed as part of the Predictive Analytics and Machine Learning using Python (PAML) module of the MSc Data Analytics programme.

The aim of the project is to use machine learning techniques to analyse the Banknote Authentication dataset and determine whether a banknote is genuine or counterfeit based on statistical measurements extracted from banknote images.

Dataset

The dataset contains four numerical features:

Variance
Skewness
Curtosis
Entropy

The target variable is Class, which indicates whether the banknote belongs to the genuine or counterfeit class.

Before modelling, the data was checked for missing values and duplicate records. Duplicate observations were removed before analysis.

Analysis and Machine Learning

The Python code covers the main stages of the project:

Data loading and inspection
Data cleaning and preparation
Exploratory data analysis (EDA)
Decision Tree classification
K-Means clustering
Linear Regression
Model evaluation and comparison
Visualisation of results
Decision Tree

A Decision Tree classifier was used to predict whether a banknote was genuine or counterfeit. Hyperparameter tuning was performed using GridSearchCV.

The model was evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion matrix
Cross-validation
K-Means Clustering

K-Means was used as an unsupervised learning method to identify groups within the banknote observations.

Different numbers of clusters were compared using the silhouette score, and the resulting clusters were also compared with the known classes.

Linear Regression

Linear Regression was used to investigate the relationship between the statistical features and Variance. Its performance was compared with a simple mean-based baseline using:

MAE
RMSE
R-squared
Residual analysis
Technologies and Libraries

The analysis was developed in Python using:

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Repository Contents
├── README.md
├── PAMPL_machine_learning_assignment.py
└── bill_authentication.csv
Purpose

This repository provides the Python implementation supporting the PAML assignment report. The code is intended to demonstrate the complete machine learning workflow from data preparation and exploratory analysis through to model development, evaluation and interpretation.

Author

Trisha Kampani
MSc Data Analytics
