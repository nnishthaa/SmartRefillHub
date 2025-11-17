
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Smart Kiosk Analytics Dashboard", layout="wide")
st.title("📊 Smart Kiosk Data Analytics Dashboard")

uploaded = st.file_uploader("Upload your survey CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())

    df_enc = df.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))

    st.sidebar.header("Filters")
    if "equipment" in df.columns:
        eq = st.sidebar.multiselect("Select Equipment", df["equipment"].unique())
        if eq:
            df = df[df["equipment"].isin(eq)]

    if "satisfaction" in df.columns:
        sat = st.sidebar.slider("Satisfaction", int(df["satisfaction"].min()), int(df["satisfaction"].max()))
        df = df[df["satisfaction"] >= sat]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Visual Insights", "🤖 ML Models", "📥 Predict New Data", "💡 Behaviour Insights"])

    # ---------------- TAB 1 FIXED CHARTS ----------------
    with tab1:
        st.subheader("Complex Charts & Insights")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Chart 1
        st.write("### Boxplot")
        x1 = st.selectbox("Choose X (categorical)", cat_cols, key="box_x")
        y1 = st.selectbox("Choose Y (numeric)", numeric_cols, key="box_y")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=x1, y=y1, ax=ax)
        st.pyplot(fig)

        # Chart 2
        st.write("### Count Plot")
        x2 = st.selectbox("Choose Category", cat_cols, key="count_x")
        hue2 = st.selectbox("Choose Hue", cat_cols, key="count_h")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=x2, hue=hue2, ax=ax)
        st.pyplot(fig)

        # Chart 3
        st.write("### Scatter Plot")
        x3 = st.selectbox("X axis (numeric)", numeric_cols, key="sc_x")
        y3 = st.selectbox("Y axis (numeric)", numeric_cols, key="sc_y")
        hue3 = st.selectbox("Hue", cat_cols, key="sc_h")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x3, y=y3, hue=hue3, ax=ax)
        st.pyplot(fig)

        # Chart 4
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(df_enc.corr(), cmap="coolwarm")
        st.pyplot(fig)

        # Chart 5
        st.write("### Bar Chart")
        x5 = st.selectbox("Bar Category", cat_cols, key="bar_x")
        y5 = st.selectbox("Bar Numeric", numeric_cols, key="bar_y")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x=x5, y=y5)
        st.pyplot(fig)

    # ---------------- TAB 2 ML MODELS ----------------
    with tab2:
        st.subheader("Run ML Algorithms")

        if st.button("Run Models"):
            target = st.selectbox("Select Target Column", df.columns)
            X = df_enc.drop(target, axis=1)
            y = df_enc[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            for name, model in models.items():
                st.write(f"### {name}")
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, pred)
                prec = precision_score(y_test, pred, zero_division=0)
                rec = recall_score(y_test, pred, zero_division=0)
                f1s = f1_score(y_test, pred, zero_division=0)

                st.write(f"Accuracy: {acc}")
                st.write(f"Precision: {prec}")
                st.write(f"Recall: {rec}")
                st.write(f"F1 Score: {f1s}")

                cm = confusion_matrix(y_test, pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, cmap="Blues")
                st.pyplot(fig)

                fpr, tpr, _ = roc_curve(y_test, prob)
                auc_score = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
                ax.plot([0,1],[0,1],'k--')
                st.pyplot(fig)

    # ---------------- TAB 3 NEW PREDICTION ----------------
    with tab3:
        st.subheader("Upload new dataset to predict")
        pred_file = st.file_uploader("Upload new CSV", type=["csv"], key="pred")
        if pred_file:
            df_new = pd.read_csv(pred_file)
            st.dataframe(df_new.head())
            st.info("Prediction model will use the previously selected ML model.")

    # ---------------- TAB 4 BEHAVIOUR INSIGHTS ----------------
    with tab4:
        st.subheader("Age Group Preference & Subscription Fee Trends")
        if "age" in df.columns and "subscription_fee" in df.columns:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x="age", y="subscription_fee")
            st.pyplot(fig)
        else:
            st.warning("Columns 'age' and 'subscription_fee' not found in dataset.")
