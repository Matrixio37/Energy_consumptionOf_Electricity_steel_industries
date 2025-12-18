import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from prophet import Prophet 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

  

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Steel Plant Energy Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBAL CONSTANTS
DATA_FILE = "steel_plant_weather_2020_2025_expanded.csv"
MODEL_FILE = "model.pkl"

# UTILITY FUNCTIONS
def show_section_header(title):
    st.markdown(f"## {title}")
    st.markdown("---")

def format_metric(value):
    return f"{value:,.2f}"

# LOAD MACHINE LEARNING MODEL
@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model

# LOAD DATASET
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

# FEATURE ENGINEERING
def add_time_features(df):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df

def add_energy_kpis(df):
    df["Energy_per_Ton"] = (
        df["Electricity_Consumption_MWh"] / df["Production_Tons"]
    ).round(3)
    return df

# DATA QUALITY CHECKS
def data_quality_report(df):
    report = {
        "Total Records": len(df),
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Date Range": f"{df['Date'].min().date()} to {df['Date'].max().date()}"
    }
    return report

# RANDOM FOREST FUTURE PREDICTION LOGIC
def predict_future_energy(model, days, scenario="Normal"):
    today = datetime.today()
    future_dates = [today + timedelta(days=i) for i in range(1, days + 1)]

    if scenario == "High Production":
        prod_range = (600, 700)
        temp_range = (30, 42)
    elif scenario == "Low Production":
        prod_range = (400, 500)
        temp_range = (20, 32)
    else:
        prod_range = (480, 620)
        temp_range = (25, 38)

    df_future = pd.DataFrame({
        "Production_Tons": np.random.uniform(*prod_range, days),
        "Temperature_Celsius": np.random.uniform(*temp_range, days),
        "Humidity_Percent": np.random.uniform(40, 80, days),
        "Weather_Condition": ["Sunny"] * days,
        "Shift": ["Morning"] * days,
        "Downtime_Hours": np.random.uniform(0, 3, days),
        "Year": [d.year for d in future_dates],
        "Month": [d.month for d in future_dates],
        "Day": [d.day for d in future_dates],
        "DayOfWeek": [d.weekday() for d in future_dates]
    })

    predictions = model.predict(df_future)

    result = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Predicted_Energy_MWh": np.round(predictions, 2)
    })

    return result

# LOAD DATA & MODEL
df = load_data()
df = add_time_features(df)
df = add_energy_kpis(df)

model = load_model()

# TITLE & INTRODUCTION
st.title("Steel Plant Electricity Consumption Analytics Dashboard")

st.markdown("""
### Project Overview

This dashboard analyzes historical electricity consumption of a steel plant and predicts future energy
requirements using **Random Forest Regression** and **Facebook Prophet**.

The goal is to assist decision-makers in:
- Understanding consumption patterns
- Identifying inefficiencies
- Planning future energy demand
""")

# SIDEBAR CONTROLS
st.sidebar.header("Control Panel")

selected_years = st.sidebar.multiselect(
    "Select Year(s)",
    options=sorted(df["Year"].unique()),
    default=sorted(df["Year"].unique())
)

selected_shifts = st.sidebar.multiselect(
    "Select Shift(s)",
    options=df["Shift"].unique(),
    default=df["Shift"].unique()
)

scenario = st.sidebar.selectbox(
    "Random Forest Scenario",
    ["Normal", "High Production", "Low Production"]
)

prediction_days = st.sidebar.slider(
    "Predict Future Days",
    min_value=5,
    max_value=30,
    value=10
)

# Apply filters
filtered_df = df[
    (df["Year"].isin(selected_years)) &
    (df["Shift"].isin(selected_shifts))
]

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Overview",
    "Trend Analysis",
    "Statistical Analysis",
    "Model Performance",
    "Future Prediction"
])

# TAB 1: DATA OVERVIEW
with tab1:
    show_section_header("Dataset Snapshot")
    st.dataframe(filtered_df.head(300))

    show_section_header("Data Quality Report")
    quality = data_quality_report(filtered_df)
    for k, v in quality.items():
        st.write(f"**{k}:** {v}")

# TAB 2: TREND ANALYSIS
with tab2:
    show_section_header("Electricity Consumption Trend")
    st.line_chart(filtered_df.set_index("Date")["Electricity_Consumption_MWh"])

    show_section_header("Production vs Consumption")
    st.line_chart(filtered_df.set_index("Date")[["Production_Tons", "Electricity_Consumption_MWh"]])

# TAB 3: STATISTICAL ANALYSIS
with tab3:
    show_section_header("Descriptive Statistics")
    st.dataframe(filtered_df.describe())

    show_section_header("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        filtered_df.select_dtypes(include=np.number).corr(),
        cmap="coolwarm",
        annot=False,
        ax=ax
    )
    st.pyplot(fig)

# TAB 4: MODEL PERFORMANCE

with tab4:
    show_section_header("Model Evaluation")

    target = "Electricity_Consumption_MWh"
    X = df.drop(["Date", target], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preds = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("MAE", format_metric(mean_absolute_error(y_test, preds)))
    col2.metric("R² Score", format_metric(r2_score(y_test, preds)))

    fig, ax = plt.subplots()
    ax.scatter(y_test, preds, alpha=0.5)
    ax.set_xlabel("Actual MWh")
    ax.set_ylabel("Predicted MWh")
    ax.set_title("Actual vs Predicted Consumption")
    st.pyplot(fig)

# TAB 5: FUTURE PREDICTION

with tab5:
    show_section_header("Future Prediction (Random Forest Scenario-Based)")
    rf_future = predict_future_energy(model, prediction_days, scenario)
    st.dataframe(rf_future)
    st.line_chart(rf_future.set_index("Date")["Predicted_Energy_MWh"])

    show_section_header("Future Prediction (Prophet Time-Series Model)")

    df_prophet = df[["Date", "Electricity_Consumption_MWh"]].copy()
    df_prophet.columns = ["ds", "y"]

    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    prophet_model.fit(df_prophet)

    future = prophet_model.make_future_dataframe(periods=prediction_days)
    forecast = prophet_model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(prediction_days)
    result.columns = ["Date", "Predicted_MWh", "Lower_Bound", "Upper_Bound"]

    st.dataframe(result)
    st.line_chart(result.set_index("Date")[["Predicted_MWh"]])
