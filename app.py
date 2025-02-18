import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import time

# Load the pre-trained model
model = joblib.load("stock_model.pkl")

# Function to fetch historical data
def fetch_historical_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

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
st.write("Enter a stock ticker and date range to analyze historical data and get real-time recommendations.")

# User input for stock ticker
ticker = st.text_input("Enter the stock ticker (e.g., RELIANCE.NS, INFY.NS):", "RELIANCE.NS")

# User input for date range
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

# Fetch historical data
if st.button("Analyze Historical Data"):
    with st.spinner("Fetching historical data..."):
        historical_data = fetch_historical_data(ticker, start_date, end_date)
        if historical_data.empty:
            st.error("No data found for the given ticker and date range.")
        else:
            st.success("Data fetched successfully!")
            st.write(f"### Historical Data for {ticker}")
            st.line_chart(historical_data['Close'])

# Real-time recommendation
if st.button("Get Real-Time Recommendation"):
    with st.spinner("Fetching real-time data..."):
        recommendation = predict_realtime(ticker)
        st.write(f"### Recommendation for {ticker}: **{recommendation}**")

        # Fetch and display real-time data
        real_time_data = fetch_real_time_data(ticker)
        if not real_time_data.empty:
            st.write("### Real-Time Stock Price")
            st.line_chart(real_time_data['Close'])

            # Highlight Buy/Sell signals
            real_time_data = create_features(real_time_data)
            real_time_data['Predicted_Signal'] = model.predict(real_time_data[['Return_1', 'Return_2', 'Return_3', 'Return_4', 'Return_5']])
            buy_signals = real_time_data[real_time_data['Predicted_Signal'] == 1]
            sell_signals = real_time_data[real_time_data['Predicted_Signal'] == -1]

            st.write("### Buy/Sell Signals")
            st.write("Green triangles (▲) indicate **Buy** signals. Red triangles (▼) indicate **Sell** signals.")
            st.line_chart(real_time_data['Close'])
            if not buy_signals.empty:
                st.write("Buy Signals:")
                st.dataframe(buy_signals[['Close', 'Predicted_Signal']])
            if not sell_signals.empty:
                st.write("Sell Signals:")
                st.dataframe(sell_signals[['Close', 'Predicted_Signal']])
