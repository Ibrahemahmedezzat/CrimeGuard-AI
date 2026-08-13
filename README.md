# 🚨 CrimeGuard AI

## Crime Risk Intelligence & Spatial Hotspot Detection

CrimeGuard AI is a machine learning-based crime intelligence
dashboard designed to predict crime risk and identify geographical
crime hotspots.

## 🎯 Project Objectives

- Predict crime risk based on location and time.
- Identify geographical crime hotspots.
- Provide an interactive crime intelligence dashboard.
- Help visualize areas with concentrated criminal activity.

## 🤖 Machine Learning Models

### Random Forest
Used for supervised crime risk classification based on:

- District
- Day
- Month
- Hour

### XGBoost
Used as an additional machine learning model for crime prediction
and model comparison.

### DBSCAN
An unsupervised clustering algorithm used to detect spatial
crime hotspots from geographical crime coordinates.

The DBSCAN analysis identified 146 spatial hotspots.

## 🗺️ Interactive Dashboard

The Streamlit dashboard provides:

- Crime risk prediction
- AI risk probability
- District / day / month / hour selection
- Interactive DBSCAN hotspot map
- Crime hotspot statistics
- Nearest police station information
- Top crime hotspots

## 🏗️ Project Structure

CrimeGuard-AI/

├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── dbscan_hotspots.pkl
└── notebooks/

## 🚀 Running Locally

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

## ⚠️ Disclaimer

This system is an academic machine learning project.
Predictions should not be interpreted as definitive crime forecasts
or used as the sole basis for real-world decisions.