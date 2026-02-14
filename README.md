# ML Assignment 2 – Bank Marketing Classification

## a. Problem Statement

The objective of this assignment is to build, evaluate, and compare multiple machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on marketing campaign data. The project also demonstrates an end-to-end machine learning workflow, including model training, evaluation, and deployment using a Streamlit web application.

---

## b. Dataset Description

The **Bank Marketing dataset** is obtained from the **UCI Machine Learning Repository**.  
It contains **45,211 instances** with **16 input features** and **1 binary target variable (`y`)**, which indicates whether the client subscribed to a term deposit (`yes` / `no`).

The dataset includes a mix of **numerical features** (such as age, campaign duration, and economic indicators) and **categorical features** (such as job, marital status, and education), making it suitable for evaluating different classification models.

---

## c. Models Used & Evaluation Metrics

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

### 🔹 Comparison Table (from notebook execution)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.891002 | 0.870741 | 0.591463 | 0.220121 | 0.320838 | 0.314978 |
| Decision Tree | 0.874104 | 0.704609 | 0.463379 | 0.483359 | 0.473158 | 0.401820 |
| kNN | 0.891799 | 0.813054 | 0.566622 | 0.318457 | 0.407748 | 0.371141 |
| Naive Bayes | 0.840927 | 0.813212 | 0.353988 | 0.436460 | 0.390921 | 0.302763 |
| Random Forest (Ensemble) | 0.906750 | 0.923778 | 0.659905 | 0.418306 | 0.512037 | 0.478078 |
| XGBoost (Ensemble) | 0.905866 | 0.927747 | 0.631902 | 0.467474 | 0.537391 | 0.493144 |

---

## d. Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieved high accuracy but low recall, indicating difficulty in identifying positive class instances in an imbalanced dataset. |
| Decision Tree | Provided balanced precision and recall but showed lower AUC, suggesting limited generalization capability. |
| kNN | Performance was sensitive to feature scaling and choice of k, resulting in moderate recall and MCC values. |
| Naive Bayes | Fast and simple model, but performance was limited due to strong feature independence assumptions. |
| Random Forest (Ensemble) | Demonstrated strong overall performance with higher AUC and MCC by reducing overfitting through ensemble learning. |
| XGBoost (Ensemble) | Achieved the best balance across all metrics, especially AUC and MCC, by effectively modeling complex non-linear relationships. |

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

---
