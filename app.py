
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

    # Encode
    df_enc = df.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))

    st.sidebar.header("Filters")
    if "equipment" in df.columns:
        equip_list = st.sidebar.multiselect("Select Equipment", df["equipment"].unique())
        if equip_list:
            df = df[df["equipment"].isin(equip_list)]

    if "satisfaction" in df.columns:
        sat = st.sidebar.slider("Satisfaction Filter", int(df["satisfaction"].min()), int(df["satisfaction"].max()))
        df = df[df["satisfaction"] >= sat]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Visual Insights", "🤖 ML Models", "📥 Predict Using New Data", "💡 Behaviour Insights"])

    with tab1:
        st.subheader("Complex Charts & Insights")

        # Chart 1
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="age", y="spending", ax=ax)
        st.pyplot(fig)

        # Chart 2
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="equipment", hue="interest", ax=ax)
        st.pyplot(fig)

        # Chart 3
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="income", y="subscription_fee", hue="interest", ax=ax)
        st.pyplot(fig)

        # Chart 4
        fig, ax = plt.subplots()
        sns.heatmap(df_enc.corr(), cmap="coolwarm")
        st.pyplot(fig)

        # Chart 5
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="age", y="subscription_fee")
        st.pyplot(fig)

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
                f1 = f1_score(y_test, pred, zero_division=0)

                st.write(f"Accuracy: {acc}")
                st.write(f"Precision: {prec}")
                st.write(f"Recall: {rec}")
                st.write(f"F1 Score: {f1}")

                # Confusion matrix
                cm = confusion_matrix(y_test, pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, cmap="Blues")
                st.pyplot(fig)

                # ROC
                fpr, tpr, _ = roc_curve(y_test, prob)
                auc_score = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
                ax.plot([0,1],[0,1],'k--')
                st.pyplot(fig)

    with tab3:
        st.subheader("Predict using new dataset")
        file2 = st.file_uploader("Upload new CSV for prediction", type=["csv"], key="newfile")
        if file2:
            df_new = pd.read_csv(file2)
            st.dataframe(df_new.head())
            st.info("Predictions functionality needs target and model selection.")

    with tab4:
        st.subheader("Age Preferences & Subscription Fee Insights")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="age", y="subscription_fee")
        st.pyplot(fig)

