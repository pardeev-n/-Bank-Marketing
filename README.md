
# Bank Marketing Classification – ML Assignment 2

## Problem Statement
The objective is to predict whether a bank customer will subscribe to a term deposit based on marketing campaign data.

## Dataset Description
The Bank Marketing dataset from the UCI Machine Learning Repository contains 41,188 records with 20 input features and one binary target variable (`y`).

## Models Used
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC) were used to compare models.

## Observations
- Logistic Regression performed well for linear relationships.
- Tree-based and ensemble models achieved higher AUC and MCC.
- XGBoost showed the best overall performance due to boosted learning.

## Streamlit App
The Streamlit application allows:
- CSV upload (test data)
- Model selection
- Metric visualization
- Confusion matrix and classification report

## Deployment
The app is deployed using Streamlit Community Cloud and connected directly to this GitHub repository.
