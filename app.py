
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import norm

# -----------------------
# Load files
# -----------------------
model = joblib.load("xgb_demand_model.pkl")
features = joblib.load("features.pkl")

history = pd.read_csv("recent_history.csv", parse_dates=["date"])
error_stats = pd.read_csv("error_stats.csv")

# -----------------------
# Page
# -----------------------
st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    layout="wide"
)

st.title("Demand Forecasting & Inventory Optimization")

st.write(
    "Predict future demand and calculate inventory recommendations."
)

# -----------------------
# User inputs
# -----------------------
store = st.selectbox(
    "Select Store",
    sorted(history["store"].unique())
)

item = st.selectbox(
    "Select Item",
    sorted(history["item"].unique())
)

current_inventory = st.number_input(
    "Current Inventory",
    min_value=0,
    value=120
)

lead_time = st.number_input(
    "Supplier Lead Time (days)",
    min_value=1,
    value=7
)

service_level = st.selectbox(
    "Service Level",
    [90, 95, 99],
    index=1
)

target_days = st.number_input(
    "Target Stock Coverage (days)",
    min_value=1,
    value=14
)

# -----------------------
# Selected product history
# -----------------------
product = history[
    (history["store"] == store) &
    (history["item"] == item)
].sort_values("date")

if len(product) < 28:
    st.error("Not enough historical data.")
    st.stop()

latest_date = product["date"].max()
forecast_date = latest_date + pd.Timedelta(days=1)

# -----------------------
# Create model features
# -----------------------
lag_1 = product["sales"].iloc[-1]
lag_7 = product["sales"].iloc[-7]
lag_28 = product["sales"].iloc[-28]

rolling_mean_7 = product["sales"].tail(7).mean()
rolling_mean_28 = product["sales"].tail(28).mean()

input_data = pd.DataFrame([{
    "store": store,
    "item": item,
    "year": forecast_date.year,
    "month": forecast_date.month,
    "day_of_week": forecast_date.dayofweek,
    "lag_1": lag_1,
    "lag_7": lag_7,
    "lag_28": lag_28,
    "rolling_mean_7": rolling_mean_7,
    "rolling_mean_28": rolling_mean_28
}])

input_data = input_data[features]

# -----------------------
# Forecast
# -----------------------
forecast = float(model.predict(input_data)[0])

# -----------------------
# Forecast error
# -----------------------
error_row = error_stats[
    (error_stats["store"] == store) &
    (error_stats["item"] == item)
]

if len(error_row) > 0:
    error_std = float(error_row["error_std"].iloc[0])
else:
    error_std = float(error_stats["error_std"].mean())

# -----------------------
# Inventory calculations
# -----------------------
z = norm.ppf(service_level / 100)

safety_stock = (
    z *
    error_std *
    np.sqrt(lead_time)
)

reorder_point = (
    forecast * lead_time
    + safety_stock
)

target_stock = (
    forecast * target_days
    + safety_stock
)

order_quantity = max(
    0,
    target_stock - current_inventory
)

# -----------------------
# Results
# -----------------------
st.subheader("Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Daily Demand Forecast",
    f"{forecast:.1f} units"
)

col2.metric(
    "Safety Stock",
    f"{safety_stock:.1f} units"
)

col3.metric(
    "Reorder Point",
    f"{reorder_point:.1f} units"
)

col4.metric(
    "Recommended Order",
    f"{order_quantity:.1f} units"
)

if current_inventory <= reorder_point:
    st.warning("⚠️ Reorder Required")
else:
    st.success("✅ Inventory Level is Sufficient")

# -----------------------
# Recent sales chart
# -----------------------
st.subheader("Recent Sales History")

chart_data = (
    product[["date", "sales"]]
    .set_index("date")
)

st.line_chart(chart_data)

st.caption(
    "Inventory recommendations use forecast uncertainty, "
    "lead time and the selected service level."
)
