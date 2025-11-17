
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Smart Kiosk Analytics Dashboard", layout="wide")
st.title("📊 Smart Kiosk Data Analytics Dashboard")

uploaded = st.file_uploader("Upload your survey CSV", type=["csv"])

def safe_classification_metrics(y_true, y_pred):
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "msg": "⚠️ Only one class present — Prec/Recall/F1 unavailable."
        }
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "msg": ""
    }

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())

    df_enc = df.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))

    st.sidebar.header("Filters")
    if len(df.columns) > 0:
        tab1, tab2 = st.tabs(["📈 Visual Insights", "🤖 ML Models"])

        with tab1:
            st.subheader("Dynamic Visual Charts")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            if len(cat_cols) > 0 and len(numeric_cols) > 0:
                x1 = st.selectbox("X (categorical)", cat_cols, key="box_x")
                y1 = st.selectbox("Y (numeric)", numeric_cols, key="box_y")
                fig, ax = plt.subplots()
                sns.boxplot(data=df, x=x1, y=y1, ax=ax)
                st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8,5))
            sns.heatmap(df_enc.corr(), cmap="coolwarm")
            st.pyplot(fig)

        with tab2:
            st.subheader("Run Classification Models")

            target = st.selectbox("Select Target Column", df.columns)

            if st.button("Run Models"):
                X = df_enc.drop(target, axis=1)
                y = df_enc[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                models = {
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier()
                }

                for name, model in models.items():
                    st.write(f"### {name}")

                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)

                    metrics = safe_classification_metrics(y_test, pred)

                    st.write(f"Accuracy: {metrics['accuracy']}")
                    st.write(f"Precision: {metrics['precision']}")
                    st.write(f"Recall: {metrics['recall']}")
                    st.write(f"F1 Score: {metrics['f1']}")
                    if metrics['msg']:
                        st.warning(metrics['msg'])

                    cm = confusion_matrix(y_test, pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, cmap="Blues")
                    st.pyplot(fig)

                    if len(np.unique(y_test)) < 2:
                        st.warning("⚠️ ROC cannot be computed — only one class present.")
                    else:
                        prob = model.predict_proba(X_test)[:,1]
                        fpr, tpr, _ = roc_curve(y_test, prob)
                        auc_score = auc(fpr, tpr)
                        fig, ax = plt.subplots()
                        ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
                        ax.plot([0,1],[0,1],'k--')
                        ax.legend()
                        st.pyplot(fig)
