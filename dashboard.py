import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load Model

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Production_energy_2021_2025.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

# Feature Engineering

def add_features(df):
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Year"] = df["Date"].dt.year
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df


# Predict Future

def predict_future(model, days=7):
    today = datetime.today()
    future_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days+1)]

    df_future = pd.DataFrame({
        "Production_Tons": np.random.uniform(450, 600, days),
        "Temperature_Celsius": np.random.uniform(25, 40, days),
        "Humidity_Percent": np.random.uniform(40, 80, days),
        "Weather_Condition": ["Sunny"] * days,
        "Shift": ["Morning"] * days,
        "Downtime_Hours": np.random.uniform(0, 3, days),
        "Month": [today.month] * days,
        "Day": list(range(1, days+1)),
        "Year": [today.year] * days,
        "DayOfWeek": list(range(1, days+1))
    })

    preds = model.predict(df_future)
    preds = [round(float(p), 2) for p in preds]

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Energy_MWh": preds
    })

# Streamlit UI

st.title(" Steel Plant Electricity Consumption Dashboard")

df = load_data()
df = add_features(df)
model = load_model()

st.sidebar.header("Dashboard Controls")
days = st.sidebar.slider("Predict next N days", 3, 30, 7)

# Show Raw Data

st.subheader(" Dataset Preview")
st.dataframe(df.head(748))

# Graphs Section

st.subheader(" Electricity Consumption Over Time")
st.line_chart(df.set_index("Date")["Electricity_Consumption_MWh"])

st.subheader(" Temperature Over Time")
st.line_chart(df.set_index("Date")["Temperature_Celsius"])

st.subheader(" Humidity Over Time")
st.line_chart(df.set_index("Date")["Humidity_Percent"])

st.subheader(" Shift-wise Electricity Consumption")
shift_avg = df.groupby("Shift")["Electricity_Consumption_MWh"].mean()
st.bar_chart(shift_avg)

st.subheader("Weather Condition Impact on Consumption")

weather_avg = df.groupby("Weather_Condition")["Electricity_Consumption_MWh"].mean()
st.bar_chart(weather_avg)



# Monthly Trend
st.subheader(" Monthly Average Electricity Consumption")
monthly_avg = df.groupby("Month")["Electricity_Consumption_MWh"].mean()
st.line_chart(monthly_avg)

# Yearly Trend
st.subheader("Yearly Electricity Consumption Trend")
yearly_avg = df.groupby("Year")["Electricity_Consumption_MWh"].mean()
st.line_chart(yearly_avg)




# Model Metrics

st.subheader(" Model Performance Metrics")

target = "Electricity_Consumption_MWh"
X = df.drop(["Date", target], axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
st.metric("R² Score", f"{r2:.3f}")


# Future Predictions

st.subheader(f" Predicted Electricity Consumption for Next {days} Days")

future_df = predict_future(model, days)
st.dataframe(future_df)

st.subheader(" Actual vs Predicted (Test Set)")

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_test, preds, alpha=0.6)
ax.set_xlabel("Actual Consumption (MWh)")
ax.set_ylabel("Predicted Consumption (MWh)")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)


# Download Button
csv = future_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Predictions as CSV", csv, "future_predictions.csv", "text/csv")
