# 📘 ML Assignment 2 – Bank Marketing Classification

## a. Problem Statement

The objective of this assignment is to build, evaluate, and compare multiple machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on marketing campaign data. The project also demonstrates an end-to-end machine learning workflow, including model training, evaluation, and deployment using a Streamlit web application.

---

## b. Dataset Description [1 Mark]

The **Bank Marketing dataset** is obtained from the **UCI Machine Learning Repository**.  
It contains **41,188 instances** with **20 input features** and **1 binary target variable (`y`)**, which indicates whether the client subscribed to a term deposit (`yes` / `no`).

The dataset includes a mix of **numerical features** (such as age, campaign duration, and economic indicators) and **categorical features** (such as job, marital status, and education), making it suitable for evaluating different classification models.

---

## c. Models Used & Evaluation Metrics [6 Marks]

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using the following metrics:
- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### 🔹 Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.90 | 0.92 | 0.63 | 0.45 | 0.52 | 0.47 |
| Decision Tree | 0.88 | 0.85 | 0.51 | 0.55 | 0.53 | 0.46 |
| kNN | 0.89 | 0.88 | 0.58 | 0.42 | 0.49 | 0.45 |
| Naive Bayes | 0.86 | 0.84 | 0.49 | 0.60 | 0.54 | 0.44 |
| Random Forest (Ensemble) | 0.91 | 0.94 | 0.66 | 0.50 | 0.57 | 0.52 |
| XGBoost (Ensemble) | 0.92 | 0.95 | 0.69 | 0.53 | 0.60 | 0.55 |

---

## d. Observations on Model Performance [3 Marks]

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Performed well for linear decision boundaries but had limited ability to capture complex feature interactions. |
| Decision Tree | Achieved reasonable accuracy but showed signs of overfitting due to its high variance nature. |
| kNN | Performance depended heavily on feature scaling and choice of neighbors, making it computationally expensive for large datasets. |
| Naive Bayes | Fast and simple model but performance was affected by the strong independence assumption between features. |
| Random Forest (Ensemble) | Improved generalization and stability by reducing overfitting through ensemble learning. |
| XGBoost (Ensemble) | Achieved the best overall performance with higher AUC and MCC by effectively capturing non-linear relationships. |

---

## Streamlit Web Application

A Streamlit web application was developed to demonstrate the trained models.  
The application includes:
- CSV dataset upload (test data only)  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix and classification report  

The app is deployed using **Streamlit Community Cloud** and connected directly to the GitHub repository.

---

## Deployment

1. Push the project code to GitHub  
2. Create a new app on Streamlit Community Cloud  
3. Select the repository and `app.py` file  
4. Deploy the application and access it via the generated public URL  
