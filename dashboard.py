#To RUN THE CODE:-  python -m streamlit run app.py



import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
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


# FUTURE PREDICTION

def predict_future_energy(model, df, days, scenario, shift):
    future_dates = pd.date_range(
        start=df["Date"].max() + timedelta(days=1),
        periods=days
    )

    base_prod = df["Production_Tons"].mean()
    base_eff = (df[TARGET] / df["Production_Tons"]).mean()

    
    factor = 1.0
    if scenario == "High Production":
        factor = 1.15
    elif scenario == "Low Production":
        factor = 0.85

    shift_factor = {
        "Morning": 1.1,
        "Afternoon": 1.0,
        "Night": 0.9
    }[shift]

    
    prod_values = np.linspace(base_prod * factor * shift_factor,
                              base_prod * factor * shift_factor * 1.05,
                              days)
    eff_values = base_eff + np.random.normal(0, base_eff*0.02, days)

    future_X = pd.DataFrame({
        "Production_Tons": prod_values,
        "Energy_per_Ton": eff_values
    })

    preds = model.predict(future_X)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Energy_MWh": preds.round(2)
    })

# SIDEBAR

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Steel Plant CSV",
    type=["csv"]
)

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Normal", "High Production", "Low Production"]
)

future_days = st.sidebar.slider(
    "Future Days",
    5, 30, 10
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

df = load_data(uploaded_file)


years = sorted(df["Date"].dt.year.unique())
year_selected = st.sidebar.selectbox(
    "Select Year",
    ["All Years"] + years
)

if year_selected != "All Years":
    df = df[df["Date"].dt.year == year_selected]


shift_selected = st.sidebar.selectbox(
    "Select Shift",
    ["Morning", "Afternoon", "Night"]
)


# LOAD MODEL

model = load_model()


# TITLE

st.title("Steel Plant Electricity Consumption Dashboard")

st.markdown(
    """
    This dashboard analyzes historical electricity consumption of a steel plant
    and predicts future energy demand using a trained ML regression model.
    """
)


# TABS

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Trends",
    "Model Performance",
    "Future Prediction"
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


# TAB 3: MODEL PERFORMANCE

with tab3:
    st.subheader("Model Evaluation")

    X = df[["Production_Tons"]].copy()
    X["Energy_per_Ton"] = df[TARGET] / df["Production_Tons"]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    preds = model.predict(X_test)

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mean_absolute_error(y_test, preds):,.2f}")
    col2.metric("R²", f"{r2_score(y_test, preds):.3f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, preds, alpha=0.6)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


# TAB 4: FUTURE PREDICTION


with tab4:
    st.subheader("Future Electricity Prediction")

    future_df = predict_future_energy(
        model,
        df,
        future_days,
        scenario,
        shift_selected
    )

    st.dataframe(ui_safe(future_df))
    st.line_chart(future_df.set_index("Date")["Predicted_Energy_MWh"])

    st.download_button(
        "Download Predictions",
        future_df.to_csv(index=False).encode("utf-8"),
        "future_predictions.csv",
        "text/csv"
    )

