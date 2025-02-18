import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
print("hello harsh vardhan soni BOSS;) LETS GO....")
# Step 1: Download historical stock data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Create features and target variable
def create_features(stock_data):
    # Use 'Close' instead of 'Adj Close' since auto_adjust=True is now the default
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()

    # Create a target variable: 1 for Buy, -1 for Sell, 0 for Hold
    stock_data['Signal'] = 0
    stock_data.loc[stock_data['Daily_Return'] > 0.01, 'Signal'] = 1  # Buy
    stock_data.loc[stock_data['Daily_Return'] < -0.01, 'Signal'] = -1  # Sell

    # Drop missing values
    stock_data.dropna(inplace=True)

    # Features: Use past 5 days' returns as features
    for i in range(1, 6):
        stock_data[f'Return_{i}'] = stock_data['Daily_Return'].shift(i)

    # Drop rows with NaN values
    stock_data.dropna(inplace=True)

    return stock_data

# Step 3: Train a machine learning model
def train_model(stock_data):
    # Define features and target
    X = stock_data[['Return_1', 'Return_2', 'Return_3', 'Return_4', 'Return_5']]
    y = stock_data['Signal']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model

# Step 4: Predict Buy/Sell signals
def predict_signals(model, stock_data):
    stock_data['Predicted_Signal'] = model.predict(stock_data[['Return_1', 'Return_2', 'Return_3', 'Return_4', 'Return_5']])
    return stock_data

# Step 5: Visualize results
def visualize_results(stock_data, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'], label='Stock Price', alpha=0.5)
    plt.scatter(stock_data[stock_data['Predicted_Signal'] == 1].index, 
                stock_data[stock_data['Predicted_Signal'] == 1]['Close'], 
                label='Buy Signal', marker='^', color='g', alpha=1)
    plt.scatter(stock_data[stock_data['Predicted_Signal'] == -1].index, 
                stock_data[stock_data['Predicted_Signal'] == -1]['Close'], 
                label='Sell Signal', marker='v', color='r', alpha=1)
    plt.title(f'{ticker} Stock Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Step 6: Get recommendation (Buy/Sell/Hold)
def get_recommendation(stock_data):
    latest_signal = stock_data['Predicted_Signal'].iloc[-1]
    if latest_signal == 1:
        return "Buy"
    elif latest_signal == -1:
        return "Sell"
    else:
        return "Hold"

# Main function
def main():
    # Get user input for stock ticker
    ticker = input("Enter the stock ticker (e.g., RELIANCE.NS, INFY.NS): ").strip().upper()

    # Get user input for start and end dates
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter the end date (YYYY-MM-DD): ").strip()

    # Download stock data
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    stock_data = download_stock_data(ticker, start_date, end_date)

    # Check if data is downloaded successfully
    if stock_data.empty:
        print(f"No data found for {ticker}. Please check the ticker symbol and dates.")
        return

    # Create features and target
    stock_data = create_features(stock_data)

    # Train the model
    print("Training the model...")
    model = train_model(stock_data)

    # Predict signals
    stock_data = predict_signals(model, stock_data)

    # Get recommendation
    recommendation = get_recommendation(stock_data)
    print(f"Recommendation for {ticker}: {recommendation}")

    # Visualize results
    visualize_results(stock_data, ticker)

if __name__ == "__main__":
    main()