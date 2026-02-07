import streamlit as st
import pandas as pd
import os
import pickle

from flaml import AutoML
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport

# BASIC CONFIG

st.set_page_config(page_title="AutoML App", layout="wide")

DATA_PATH = "dataset.csv"
MODEL_PATH = "best_model.pkl"

# LOAD DATA IF EXISTS

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = None

# SIDEBAR

with st.sidebar:
    st.title("AutoML Pipeline")
    choice = st.radio(
        "Navigation",
        ["Upload", "Profiling", "Modelling", "Download"]
    )
    st.info("Upload → Profile → Train → Download")

# UPLOAD

if choice == "Upload":
    st.header("Upload Dataset")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        df.to_csv(DATA_PATH, index=False)
        st.success("Dataset uploaded successfully.")
        st.dataframe(df)


# PROFILING

if choice == "Profiling":
    st.header("Exploratory Data Analysis")

    if df is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    with st.spinner("Generating profiling report..."):
        profile = ProfileReport(
            df,
            explorative=True,
            minimal=True
        )

    st.components.v1.html(
        profile.to_html(),
        height=900,
        scrolling=True
    )

# MODELLING

if choice == "Modelling":
    st.header("AutoML Modelling (FLAML)")

    if df is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    target = st.selectbox("Select target column", df.columns)

    time_budget = st.slider(
        "Time budget (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        step=30
    )

    if st.button("Run AutoML"):

        # ---------- CLEAN DATA ----------
        df_clean = df.copy()

        # Replace inf / -inf
        df_clean = df_clean.replace([float("inf"), float("-inf")], pd.NA)

        # Drop rows with invalid target
        df_clean = df_clean.dropna(subset=[target])

        # Drop rows with invalid features
        df_clean = df_clean.dropna()

        if df_clean.shape[0] < 20:
            st.error("Not enough clean rows to train a model.")
            st.stop()

        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        automl = AutoML()

        with st.spinner("Training models..."):
            automl.fit(
                X_train,
                y_train,
                task="regression",
                time_budget=time_budget,
                n_jobs=1,
                estimator_list=["lgbm", "rf"]  # stable on Windows
            )

        st.success("Training completed.")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(automl, f)

        st.subheader("Best Model")
        st.write(automl.model)

        st.subheader("Best Validation Loss")
        st.write(automl.best_loss)

# --------------------------------------------------
# DOWNLOAD

if choice == "Download":
    st.header("Download Trained Model")

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            st.download_button(
                label="Download Model (.pkl)",
                data=f,
                file_name="best_model.pkl"
            )
    else:
        st.warning("No trained model found. Train a model first.")




