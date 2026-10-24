Demand Forecasting & Inventory Optimization

An end-to-end machine learning project that forecasts retail demand and converts predictions into practical inventory decisions.

Overview

This project uses historical sales data to predict future demand and support inventory planning.

The system combines:

Time-series feature engineering

XGBoost regression

Forecast evaluation

Inventory optimization

Streamlit dashboard

Docker containerization

Dataset

The dataset contains approximately 913,000 daily sales records across:

10 stores

50 items

2013–2017

Main columns:

date

store

item

sales

Feature Engineering

The model uses:

Lag 1, Lag 7, Lag 28

7-day rolling average

28-day rolling average

Month

Year

Day of week

Store and item information

Model

The main model is an XGBoost Regressor.

A 7-day lag forecast was used as the baseline.

Performance

Baseline MAE: 9.08

XGBoost MAE: 6.22

Baseline RMSE: 12.05

XGBoost RMSE: 8.07

MAE improvement: 31.47%

Average Time-Series CV MAE: 6.31

Inventory Optimization

The forecast is converted into inventory recommendations using:

Safety Stock

Reorder Point

Recommended Order Quantity

Service Level comparison

The system supports service levels such as 90%, 95%, and 99%.

Higher service levels reduce stockout risk but require more inventory.

Streamlit Dashboard

The interactive dashboard allows users to choose:

Store

Item

Current Inventory

Supplier Lead Time

Service Level

Target Stock Coverage

The dashboard returns:

Daily Demand Forecast

Safety Stock

Reorder Point

Recommended Order Quantity

Reorder Decision

Recent Sales History

Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, SciPy, Streamlit, Joblib, Docker

Project Structure

├── app.py
├── xgb_demand_model.pkl
├── features.pkl
├── recent_history.csv
├── error_stats.csv
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore

Business Value

This project goes beyond demand prediction by translating forecasts into inventory decisions.

It helps answer:

When should inventory be reordered?

How much safety stock should be maintained?

How much inventory should be ordered?

How does service level affect inventory requirements?

Future Improvements

Promotion and holiday features

Product pricing

Supplier-specific lead times

Holding and stockout costs

Automated retraining

Cloud deployment and monitoring
