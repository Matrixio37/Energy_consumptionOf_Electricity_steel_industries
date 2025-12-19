# to run this app, use -  python -m streamlit run dashboard.py

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

# BACKTEST (ACCURACY)
def backtest_accuracy(model, df, days):
    df_bt = df.copy()
    df_bt["Energy_per_Ton"] = df_bt[TARGET] / df_bt["Production_Tons"]

    train_df = df_bt.iloc[:-days]
    test_df = df_bt.iloc[-days:]

    X_train = train_df[["Production_Tons", "Energy_per_Ton"]]
    y_train = train_df[TARGET]

    X_test = test_df[["Production_Tons", "Energy_per_Ton"]]
    y_test = test_df[TARGET]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    accuracy = 100 - mape

    result_df = test_df[["Date", TARGET]].copy()
    result_df["Predicted_Energy_MWh"] = preds.round(2)

    return result_df, mae, r2, accuracy

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

# APPLY SHIFT FILTER
if shift_selected != "All Shifts" and "Shift" in df.columns:
    df = df[df["Shift"] == shift_selected]

# YEAR FILTER
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
    and evaluates model accuracy using actual vs predicted values.
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

    st.subheader("Energy Efficiency (MWh per Ton)")
    df["Energy_per_Ton"] = df[TARGET] / df["Production_Tons"]
    st.line_chart(df.set_index("Date")["Energy_per_Ton"])

    st.subheader("Temperature Trend")

    if "Temperature_Celsius" in df.columns:
        fig_temp, ax_temp = plt.subplots(figsize=(10, 4))
        ax_temp.plot(df["Date"], df["Temperature_Celsius"], marker="o")
        ax_temp.set_xlabel("Date")
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.set_title("Temperature Over Time")
        fig_temp.autofmt_xdate(rotation=45)
        st.pyplot(fig_temp)
    else:
        st.warning("Temperature_Celsius column not found in dataset.")

# TAB 3: MODEL PERFORMANCE
with tab3:
    st.subheader("Model Performance (Hold-out Test Set)")

    X = df[["Production_Tons"]].copy()
    X["Energy_per_Ton"] = df[TARGET] / df["Production_Tons"]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    preds = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("MAE (MWh)", f"{mean_absolute_error(y_test, preds):.2f}")
    col2.metric("R²", f"{r2_score(y_test, preds):.3f}")

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(y_test, preds, alpha=0.6)
    ax1.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )
    ax1.set_xlabel("Actual Electricity (MWh)")
    ax1.set_ylabel("Predicted Electricity (MWh)")
    ax1.set_title("Actual vs Predicted")
    st.pyplot(fig1)

# TAB 4: PREDICTION ACCURACY
with tab4:
    st.subheader("Prediction Accuracy (Recent Actual Data)")

    backtest_days = st.slider(
        "Validation Window (Days)",
        5, min(30, len(df) - 1), 10
    )

    bt_df, mae, r2, acc = backtest_accuracy(
        model, df, backtest_days
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (MWh)", f"{mae:.2f}")
    c2.metric("R²", f"{r2:.3f}")
    c3.metric("Accuracy (%)", f"{acc:.2f}")

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
