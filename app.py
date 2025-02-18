import streamlit as st
import yfinance as yf
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load("stock_model.pkl")

# Function to fetch real-time data
def fetch_real_time_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")
    return data

# Function to create features
def create_features(data):
    data['Daily_Return'] = data['Close'].pct_change()
    for i in range(1, 6):
        data[f'Return_{i}'] = data['Daily_Return'].shift(i)
    data.dropna(inplace=True)
    return data

# Function to predict Buy/Sell/Hold
def predict_realtime(ticker):
    data = fetch_real_time_data(ticker)
    data = create_features(data)
    if not data.empty:
        latest_data = data.iloc[-1][['Return_1', 'Return_2', 'Return_3', 'Return_4', 'Return_5']].values.reshape(1, -1)
        prediction = model.predict(latest_data)
        if prediction == 1:
            return "Buy"
        elif prediction == -1:
            return "Sell"
        else:
            return "Hold"

# Streamlit app
st.title("Real-Time Stock Buy/Sell Predictor")
st.write("Enter a stock ticker to get a Buy/Sell recommendation.")

# User input for stock ticker
ticker = st.text_input("Enter the stock ticker (e.g., RELIANCE.NS, INFY.NS):", "RELIANCE.NS")

# Button to get recommendation
if st.button("Get Recommendation"):
    recommendation = predict_realtime(ticker)
    st.write(f"Recommendation for **{ticker}**: **{recommendation}**")