import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Download historical stock data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Create features and target variable
def create_features(stock_data):
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Signal'] = 0
    stock_data.loc[stock_data['Daily_Return'] > 0.01, 'Signal'] = 1  # Buy
    stock_data.loc[stock_data['Daily_Return'] < -0.01, 'Signal'] = -1  # Sell
    stock_data.dropna(inplace=True)

    # Features: Use past 5 days' returns as features
    for i in range(1, 6):
        stock_data[f'Return_{i}'] = stock_data['Daily_Return'].shift(i)
    stock_data.dropna(inplace=True)

    return stock_data

# Step 3: Train and save the model
def train_and_save_model(ticker, start_date, end_date, model_filename):
    # Download data
    stock_data = download_stock_data(ticker, start_date, end_date)

    # Create features
    stock_data = create_features(stock_data)

    # Define features and target
    X = stock_data[['Return_1', 'Return_2', 'Return_3', 'Return_4', 'Return_5']]
    y = stock_data['Signal']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model to a file
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

# Main function
if __name__ == "__main__":
    ticker = "RELIANCE.NS"  # Example: Reliance Industries on NSE
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    model_filename = "stock_model.pkl"  # File to save the model
    train_and_save_model(ticker, start_date, end_date, model_filename)