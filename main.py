import json
import re
import pandas as pd
import cohere
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# Initialize Cohere client
co = cohere.Client(st.secrets.get("COHERE_API_KEY", None))

# === Stock-related functions ===

def get_stock_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y')
        if data.empty:
            return f"Error: No data found for ticker '{ticker}'. Please check the symbol."
        return f"The current stock price of {ticker} is ${data.iloc[-1].Close:.2f}"
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"

def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    sma = data.rolling(window=window).mean().iloc[-1]
    return f"The {window}-day SMA for {ticker} is ${sma:.2f}"

def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    ema = data.ewm(span=window, adjust=False).mean().iloc[-1]
    return f"The {window}-day EMA for {ticker} is ${ema:.2f}"

def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    return f"The RSI for {ticker} is {rsi:.2f}"

def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    macd = short_EMA - long_EMA
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return f"The MACD for {ticker} is {macd.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}, Histogram: {histogram.iloc[-1]:.2f}"

def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data.Close, label='Close Price')
    ax.set_title(f'{ticker} Stock Price Over Last Year')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)
    ax.legend()
    return fig

# === Command Parsing ===

def parse_user_command(input_text):
    patterns = {
        'get_stock_price': r"(?i)stock price of (\w+)",
        'calculate_SMA': r"(?i)(\d+)[- ]?day SMA of (\w+)",
        'calculate_EMA': r"(?i)(\d+)[- ]?day EMA of (\w+)",
        'calculate_RSI': r"(?i)RSI of (\w+)",
        'calculate_MACD': r"(?i)MACD (?:for|of) (\w+)",
        'plot_stock_price': r"(?i)plot.*?(\w+)"
    }

    for func, pattern in patterns.items():
        match = re.search(pattern, input_text)
        if match:
            if func in ['calculate_SMA', 'calculate_EMA']:
                return func, {'window': int(match.group(1)), 'ticker': match.group(2).upper()}
            else:
                return func, {'ticker': match.group(1).upper()}
    return None, None

# === Streamlit App UI ===

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title('📊 Stock Analysis Chatbot Assistant')

example_questions = [
    "What’s the stock price of AAPL?",
    "What is the 50-day SMA of AAPL?",
    "What is the RSI of AMZN?",
    "Show me the MACD for META.",
    "Plot the stock price of AAPL."
]

user_input = st.text_input('Your input:')

st.markdown("### 💡 Try one of these:")
for question in example_questions:
    if st.button(question):
        user_input = question

# === Main Logic ===

if user_input:
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    func_name, args = parse_user_command(user_input)

    if func_name:
        if func_name == 'plot_stock_price':
            fig = plot_stock_price(**args)
            st.pyplot(fig)  # Directly render the plot
            st.success(f"Here's the stock price chart for {args['ticker']}.")
        else:
            result = globals()[func_name](**args)
            st.success(result)
    else:
        # fallback to Cohere chat
        response = co.chat(
            message=user_input,
            conversation_id="stock-chat",
            temperature=0.5
        )
        st.success(response.text)
        st.session_state['messages'].append({'role': 'assistant', 'content': response.text})
