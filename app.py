import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Cross Validation App", layout="centered")

st.title("📊 Cross Validation App")

uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=["csv"],
    help="Limit 200MB per file • CSV"
)

if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        k = st.slider("K Folds", 2, 10, 5)

        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=KFold(k), scoring="r2")

        st.write("Scores:", scores)
        st.write("Mean R²:", np.mean(scores))
