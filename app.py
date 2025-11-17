
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Smart Kiosk Dashboard", layout="wide")
st.title("📊 Smart Kiosk Data Analytics Dashboard")

uploaded = st.file_uploader("Upload your survey CSV", type=["csv"])

# SAFE METRICS FIX
def safe_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return acc, 0, 0, 0, "⚠️ Only one class present — some metrics unavailable."
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1, ""

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    df_enc = df.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))

    st.sidebar.header("Filters")
    if "equipment" in df.columns:
        eq = st.sidebar.multiselect("Equipment", df["equipment"].unique())
        if eq:
            df = df[df["equipment"].isin(eq)]

    if "satisfaction" in df.columns:
        sat = st.sidebar.slider("Satisfaction", int(df["satisfaction"].min()), int(df["satisfaction"].max()))
        df = df[df["satisfaction"] >= sat]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visual Insights",
        "🤖 ML Classification",
        "📥 Predict New Data",
        "💡 Behaviour Insights"
    ])

    # ==== TAB 1 ====
    with tab1:
        st.subheader("5 Complex Charts")

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if num_cols and cat_cols:
            st.write("### Chart 1: Boxplot")
            x1 = st.selectbox("X (categorical)", cat_cols)
            y1 = st.selectbox("Y (numeric)", num_cols)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=x1, y=y1, ax=ax)
            st.pyplot(fig)

        if cat_cols:
            st.write("### Chart 2: Count Plot")
            cx = st.selectbox("Category", cat_cols, key="c1")
            ch = st.selectbox("Hue", cat_cols, key="c2")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=cx, hue=ch, ax=ax)
            st.pyplot(fig)

        if len(num_cols) >= 2:
            st.write("### Chart 3: Scatter Plot")
            sx = st.selectbox("X numeric", num_cols, key="s1")
            sy = st.selectbox("Y numeric", num_cols, key="s2")
            hue = st.selectbox("Hue", cat_cols, key="s3") if cat_cols else None
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=sx, y=sy, hue=hue, ax=ax)
            st.pyplot(fig)

        st.write("### Chart 4: Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(df_enc.corr(), cmap="coolwarm")
        st.pyplot(fig)

        if cat_cols and num_cols:
            st.write("### Chart 5: Bar Chart")
            bx = st.selectbox("Bar Category", cat_cols, key="b1")
            by = st.selectbox("Bar Numeric", num_cols, key="b2")
            fig, ax = plt.subplots()
            sns.barplot(data=df, x=bx, y=by)
            st.pyplot(fig)

    # ==== TAB 2 ====
    with tab2:
        st.subheader("Run 3 Classification Algorithms")

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

            auc_data = {}

            for name, model in models.items():
                st.write(f"## {name}")
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                acc, prec, rec, f1, msg = safe_metrics(y_test, pred)
                st.write(f"Accuracy: {acc}")
                st.write(f"Precision: {prec}")
                st.write(f"Recall: {rec}")
                st.write(f"F1 Score: {f1}")
                if msg:
                    st.warning(msg)

                cm = confusion_matrix(y_test, pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, cmap="Blues")
                st.pyplot(fig)

                if len(np.unique(y_test)) > 1 and hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, prob)
                    auc_score = auc(fpr, tpr)
                    auc_data[name] = (fpr, tpr, auc_score)

                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ ROC cannot be computed.")

            if auc_data:
                st.write("### Combined AUC Plot")
                fig, ax = plt.subplots()
                for name, (fpr, tpr, a) in auc_data.items():
                    ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})")
                ax.plot([0, 1], [0, 1], 'k--')
                ax.legend()
                st.pyplot(fig)

    # ==== TAB 3 ====
    with tab3:
        st.subheader("Upload New Dataset for Prediction")
        newfile = st.file_uploader("Upload new CSV", type=["csv"], key="newcsv")
        if newfile:
            df_new = pd.read_csv(newfile)
            st.dataframe(df_new.head())

    # ==== TAB 4 ====
    with tab4:
        st.subheader("Age vs Subscription Fee")
        if "age" in df.columns and "subscription_fee" in df.columns:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x="age", y="subscription_fee")
            st.pyplot(fig)
        else:
            st.warning("age and subscription_fee columns required.")
