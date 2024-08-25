import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import ta

# Streamlit app
st.title('Enhanced Stock Price Prediction')

# User input for ticker symbol
ticker = st.text_input('Enter Stock Ticker Symbol', 'NVDA')

# Fetch stock data
data = yf.download(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))

# Check if 'Close' column exists
if 'Close' not in data.columns:
    st.error("The 'Close' column is missing from the data.")
else:
    # Prepare the data
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

    # Add technical indicators
    data['SMA'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd(data['Close'])
    data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])
    data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Features and target
    features = ['Date', 'SMA', 'EMA', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low']
    X = data[features]
    y = data['Close']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Prepare future dates for prediction
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 4)]
    future_dates_ordinal = [pd.Timestamp(date).toordinal() for date in future_dates]

    # Create a DataFrame for future dates
    future_data = pd.DataFrame({
        'Date': future_dates_ordinal,
        'SMA': [data['SMA'].iloc[-1]] * 3,
        'EMA': [data['EMA'].iloc[-1]] * 3,
        'RSI': [data['RSI'].iloc[-1]] * 3,
        'MACD': [data['MACD'].iloc[-1]] * 3,
        'Bollinger_High': [data['Bollinger_High'].iloc[-1]] * 3,
        'Bollinger_Low': [data['Bollinger_Low'].iloc[-1]] * 3
    })

    # Make predictions for the next 3 business days
    future_predictions = model.predict(future_data)

    # Convert ordinal dates back to datetime
    future_data['Date'] = future_data['Date'].map(pd.Timestamp.fromordinal)
    future_data.set_index('Date', inplace=True)

    # Combine actual and predicted data for plotting
    data['Predicted'] = np.nan
    data.loc[X_test.index, 'Predicted'] = predictions

    # Calculate gains with error handling
    ytd_gain = (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100 if len(data) > 0 else np.nan
    last_week_gain = (data['Close'][-1] - data['Close'][-5]) / data['Close'][-5] * 100 if len(data) > 5 else np.nan
    last_month_gain = (data['Close'][-1] - data['Close'][-21]) / data['Close'][-21] * 100 if len(data) > 21 else np.nan
    last_year_gain = (data['Close'][-1] - data['Close'][-252]) / data['Close'][-252] * 100 if len(data) > 252 else np.nan

    # Fetch past week's stock prices
    past_week_data = data.tail(7)

    # Display the results
    st.write(f"Next 3 Business Days Forecast for {ticker}:")
    st.write(future_data[['Close']])

    st.write(f"Past Week's Stock Prices for {ticker}:")
    st.write(past_week_data[['Close']])

    st.write(f"Year-to-Date Gain: {ytd_gain:.2f}%")
    st.write(f"Last Week Gain: {last_week_gain:.2f}%")
    st.write(f"Last Month Gain: {last_month_gain:.2f}%")
    st.write(f"Last Year Gain: {last_year_gain:.2f}%")

    # Plot the results using Streamlit
    st.line_chart(data[['Close', 'Predicted']])
    st.line_chart(future_data['Close'])