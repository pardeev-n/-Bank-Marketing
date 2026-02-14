import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing Classification", layout="wide")

st.title("📊 Bank Marketing – Classification Models")

st.markdown("Upload **test dataset only** and select a trained model to evaluate performance.")

# Load models
models = {
    "Logistic Regression": joblib.load("model/logistic.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

scaler = joblib.load("model/scaler.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
       df[col] = df[col].astype('category').cat.codes

    X = df.drop('y', axis=1)
    y = df['y']

    
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    if model_name in ["Logistic Regression", "KNN"]:
        X = scaler.transform(X)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.subheader("📈 Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col2.metric("Precision", round(precision_score(y, y_pred), 4))
    col3.metric("Recall", round(recall_score(y, y_pred), 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y, y_pred), 4))
    col5.metric("AUC", round(roc_auc_score(y, y_prob), 4))
    col6.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
