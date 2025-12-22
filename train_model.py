import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("steel_plant.csv")  


df = df.dropna(subset=["Production_Tons", "Electricity_Consumption_MWh"])


X = df[["Production_Tons"]]


y = df["Electricity_Consumption_MWh"]


model = RandomForestRegressor(
    n_estimators=100,
    random_state=20,
    n_jobs=-1
)

model.fit(X, y)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")
    
