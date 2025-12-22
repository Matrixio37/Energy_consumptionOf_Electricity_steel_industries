#Run Using :- python -m streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


# PAGE CONFIG

st.set_page_config(
    page_title="Steel Plant Energy Analytics",
    layout="wide"
)

MODEL_FILE = "model.pkl"
TARGET = "Electricity_Consumption_MWh"


# UI SAFE FUNCTION

def ui_safe(df):
    df = df.copy()
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    return df


# LOAD MODEL 

@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


# LOAD DATA

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    return df


# BACKTEST ACCURACY 

def backtest_accuracy(model, df, days):
    test_df = df.iloc[-days:].copy()

  
    X_test = test_df[["Production_Tons"]]
    y_test = test_df[TARGET]

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    
    smape = np.mean(
        2 * np.abs(preds - y_test) /
        (np.abs(y_test) + np.abs(preds))
    ) * 100

    accuracy = 100 - smape

    tolerance = 0.05
    tolerance_accuracy = (
        (np.abs(y_test - preds) / y_test) <= tolerance
    ).mean() * 100

    result_df = test_df[["Date", TARGET]].copy()
    result_df["Predicted_Energy_MWh"] = preds.round(2)

    return result_df, mae, r2, accuracy, tolerance_accuracy


# SIDEBAR CONTROLS

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Steel Plant CSV",
    type=["csv"]
)

shift_selected = st.sidebar.selectbox(
    "Select Shift",
    ["All Shifts", "Morning", "Evening", "Night"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

df = load_data(uploaded_file)


if shift_selected != "All Shifts" and "Shift" in df.columns:
    df = df[df["Shift"] == shift_selected]


years = sorted(df["Date"].dt.year.unique())
year_selected = st.sidebar.selectbox(
    "Select Year",
    ["All Years"] + list(years)
)

if year_selected != "All Years":
    df = df[df["Date"].dt.year == year_selected]

model = load_model()

# TITLE

st.title("Steel Plant Electricity Consumption Dashboard")

st.markdown(
    """
    This dashboard analyzes historical electricity consumption
    and evaluates prediction accuracy using regression metrics.
    """
)


# TABS

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Trends",
    "Model Performance",
    "Prediction Accuracy"
])


# TAB 1: DATA OVERVIEW

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(ui_safe(df.head(200)))


# TAB 2: TRENDS

with tab2:
    st.subheader("Electricity Consumption Trend")
    st.line_chart(df.set_index("Date")[TARGET])

    st.subheader("Production vs Consumption")
    st.line_chart(df.set_index("Date")[["Production_Tons", TARGET]])

    st.subheader("Temperature Trend")
    if "Temperature_C" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["Date"], df["Temperature_C"], marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Temperature Over Time")
        fig.autofmt_xdate(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Temperature_C column not found.")


# TAB 3: MODEL PERFORMANCE

with tab3:
    st.subheader("Model Performance (Hold-out Test Set)")

    X = df[["Production_Tons"]]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    preds = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("MAE (MWh)", f"{mean_absolute_error(y_test, preds):.2f}")
    col2.metric("R²", f"{r2_score(y_test, preds):.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, preds, alpha=0.6)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )
    ax.set_xlabel("Actual Electricity (MWh)")
    ax.set_ylabel("Predicted Electricity (MWh)")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


# TAB 4: PREDICTION ACCURACY

with tab4:
    st.subheader("Prediction Accuracy (Recent Data)")

    backtest_days = st.slider(
        "Validation Window (Days)",
        10, min(60, len(df) - 1), 30
    )

    bt_df, mae, r2, acc, tol_acc = backtest_accuracy(
        model, df, backtest_days
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (MWh)", f"{mae:.2f}")
    c2.metric("R²", f"{r2:.3f}")
    c3.metric("Accuracy (%)", f"{acc:.2f}")
    c4.metric("Within ±5%", f"{tol_acc:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bt_df["Date"], bt_df[TARGET], label="Actual", marker="o")
    ax.plot(
        bt_df["Date"],
        bt_df["Predicted_Energy_MWh"],
        label="Predicted",
        marker="x"
    )
    ax.set_title("Actual vs Predicted (Validation Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Electricity (MWh)")
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    st.pyplot(fig)

    st.dataframe(ui_safe(bt_df))

    st.download_button(
        label="Download Predicted Values (CSV)",
        data=bt_df.to_csv(index=False).encode("utf-8"),
        file_name="predicted_vs_actual_energy.csv",
        mime="text/csv"
    )
