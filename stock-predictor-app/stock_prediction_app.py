import streamlit as st
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sentencepiece  # Ensure sentencepiece is installed

# Load the Llama3 model and tokenizer
model_name = "C:\\Users\\tipot\\.ollama\\models"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate predictions
def generate_prediction(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

# Streamlit app
st.title("Stock Prediction App using Llama3")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker", "AAPL")

# Fetch stock data
data = yf.download(ticker, start="2024-08-18", end="2024-08-21")
st.subheader("Stock Data")
st.write(data.tail())

# User input for prediction prompt
prompt = st.text_area("Enter Prediction Prompt", "Predict the stock price for the next 30 days.")

# Generate prediction
if st.button("Generate Prediction"):
    prediction = generate_prediction(prompt)
    st.subheader("Prediction")
    st.write(prediction)
