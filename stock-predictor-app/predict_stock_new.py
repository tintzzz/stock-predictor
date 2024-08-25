import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Streamlit app
st.title('Stock Price Prediction')

# User input for ticker symbol
ticker = st.text_input('Enter Stock Ticker Symbol', 'NVDA')

# Fetch stock data
data = yf.download(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))

# Prepare the data
data['Date'] = data.index
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

# Features and target
X = data[['Date']]
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

# Make predictions for the next 3 business days
future_predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1))

# Combine actual and predicted data for plotting
data['Predicted'] = np.nan
data.loc[X_test.index, 'Predicted'] = predictions

future_data = pd.DataFrame({
    'Date': future_dates,
    'Close': future_predictions
})
future_data.set_index('Date', inplace=True)

# Calculate gains
ytd_gain = (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100
last_week_gain = (data['Close'][-1] - data['Close'][-5]) / data['Close'][-5] * 100
last_month_gain = (data['Close'][-1] - data['Close'][-21]) / data['Close'][-21] * 100
last_year_gain = (data['Close'][-1] - data['Close'][-252]) / data['Close'][-252] * 100

# Fetch past week's stock prices
past_week_data = data.tail(7)

# Display the results
st.write(f"Next 3 Business Days Forecast for {ticker}:")
st.write(future_data)

st.write(f"Past Week's Stock Prices for {ticker}:")
st.write(past_week_data[['Close']])

st.write(f"Year-to-Date Gain: {ytd_gain:.2f}%")
st.write(f"Last Week Gain: {last_week_gain:.2f}%")
st.write(f"Last Month Gain: {last_month_gain:.2f}%")
st.write(f"Last Year Gain: {last_year_gain:.2f}%")

# Plot the results using Streamlit
st.line_chart(data[['Close', 'Predicted']])
st.line_chart(future_data['Close'])