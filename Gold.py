import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json
import os
from datetime import datetime, date

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# --- Configuration & Constants ---
DATA_FILE = 'users.json'
GOLD_TICKER = 'GC=F'
USD_INR_TICKER = 'INR=X'

# CONVERSION LOGIC:
# GC=F is USD per 1 Troy Ounce.
# 1 Troy Ounce = 31.1035 Grams.
GRAMS_IN_OUNCE = 31.1035

# --- Page Config & Dark Theme CSS ---
st.set_page_config(page_title="Gold Price Dashboard (India)", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Dark Theme Colors */
    .stApp {
        background-color: #0e1117;
        color: #e1e1e1;
    }
    .stSidebar {
        background-color: #161b22;
    }
    
    /* Clean Metric Styling */
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
    }
    
    .stMetric > label {
        color: #8b949e !important;
        font-size: 18px;
        font-weight: 600;
    }
    .stMetric > div > data {
        color: #FFD700 !important; /* Gold color for values */
        font-size: 32px;
        font-weight: bold;
    }
    
    /* Insight Box Styling */
    .insight-box {
        background-color: #21262d;
        border-left: 5px solid #FFD700;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #FFD700 !important;
        border-bottom: 2px solid #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Authentication Module ---

def load_users():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(DATA_FILE, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    users = load_users()
    hashed_pwd = hash_password(password)
    return username in users and users[username]['password'] == hashed_pwd

def register_user(username, password, name):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        'password': hash_password(password),
        'name': name,
        'date_joined': str(date.today())
    }
    save_users(users)
    return True, "User registered successfully!"

# --- 2. Helper Functions ---

def format_inr(value):
    """Formats number in Indian Rupee style with commas"""
    return f"₹{value:,.2f}"

def get_usd_inr_rate():
    """Fetch current USD to INR conversion rate"""
    try:
        ticker = yf.Ticker(USD_INR_TICKER)
        data = ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        pass
    return 83.0 # Fallback

@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return pd.DataFrame()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Close']].copy()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        df = df.asfreq('D')
        df = df.ffill()
        
        # --- CRITICAL CONVERSION LOGIC ---
        # 1. Convert USD to INR
        conversion_rate = get_usd_inr_rate()
        df['Price_INR_Ounce'] = df['Close'] * conversion_rate
        
        # 2. Convert Per Ounce to Per Gram
        df['Price_PER_GRAM'] = df['Price_INR_Ounce'] / GRAMS_IN_OUNCE
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna(), autolag='AIC')
    p_value = result[1]
    return p_value, p_value < 0.05

def get_moving_averages(df):
    df['MA_30'] = df['Price_PER_GRAM'].rolling(window=30).mean()
    df['MA_90'] = df['Price_PER_GRAM'].rolling(window=90).mean()
    return df

def train_arima_model(train_data, order=(5,1,0)):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

# --- 3. UI Components ---

def show_login_ui():
    st.title("🔑 Gold Price Prediction System")
    st.markdown("### Please Login or Signup to Continue")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if authenticate(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

    with tab2:
        with st.form("signup_form"):
            new_name = st.text_input("Full Name")
            new_user = st.text_input("Create Username")
            new_pass = st.text_input("Create Password", type="password")
            submit_reg = st.form_submit_button("Sign Up")
            
            if submit_reg:
                if not new_user or not new_pass:
                    st.warning("Please fill all fields.")
                else:
                    success, msg = register_user(new_user, new_pass, new_name)
                    if success:
                        st.success(msg + " Please login now.")
                    else:
                        st.error(msg)

def show_main_app():
    # --- Sidebar ---
    st.sidebar.title(f"👋 {st.session_state.get('username', 'User')}")
    st.sidebar.markdown("---")
    
    forecast_days = st.sidebar.slider("📅 Forecast Days", 1, 365, 30)
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2015-01-01'))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    # --- Main Title ---
    st.title("🪙 Gold Price Prediction Dashboard (India)")
    st.markdown("##### Real-time analysis & forecasting per gram in INR")

    # --- Load Data ---
    with st.spinner("Fetching live data..."):
        df = load_data(GOLD_TICKER, start_date, end_date)
    
    if df.empty:
        st.error("No data found.")
        return

    df = get_moving_averages(df)
    
    # Create display dataframe
    display_df = df[['Price_PER_GRAM', 'MA_30', 'MA_90']].copy()
    display_df.rename(columns={
        'Price_PER_GRAM': 'Gold Price (₹/g)',
        'MA_30': '30-Day Avg',
        'MA_90': '90-Day Avg'
    }, inplace=True)

    # --- Insights Section (Top Metrics) ---
    latest_price = df['Price_PER_GRAM'].iloc[-1]
    ma_30 = df['MA_30'].iloc[-1]
    ma_90 = df['MA_90'].iloc[-1]
    
    short_trend = "Increasing 📈" if latest_price > ma_30 else "Decreasing 📉"
    long_trend = "Increasing 📈" if latest_price > ma_90 else "Decreasing 📉"

    # Using native Streamlit Metrics (Cleaner than custom HTML cards)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Current Price (Per Gram)", format_inr(latest_price))
        
    with col2:
        st.metric("📊 Short-Term Trend (30D)", short_trend, delta=f"{((latest_price - ma_30)/ma_30)*100:.2f}%")
        
    with col3:
        st.metric("📉 Long-Term Trend (90D)", long_trend, delta=f"{((latest_price - ma_90)/ma_90)*100:.2f}%")

    # Simple Insight Message
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    if latest_price > ma_30:
        st.success(f"✅ **Insight:** Gold price is currently **higher** than the 30-day average. The short-term trend is positive.")
    else:
        st.warning(f"⚠️ **Insight:** Gold price is currently **lower** than the 30-day average. The market is in a short-term dip.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # --- Beginner Explanation Box ---
    with st.expander("🎓 New to Gold Trading? Read this!"):
        st.markdown("""
        * **What is Price per Gram?** We converted the international market price to Indian Rupees per gram for easy understanding.
        * **30-Day Average:** Average price over the last month. Current price above this means price is going up recently.
        * **90-Day Average:** Average price over the last 3 months. Helps understand the long-term direction.
        """)

    # --- Tabs (Removed AI Model Tab) ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "⬇️ Download"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📈 Gold Price History (₹ per gram)")
            st.line_chart(display_df, color=['#FFD700', '#1E90FF', '#FF6347'])
            
        with c2:
            st.subheader("🔢 Statistics")
            stats = df['Price_PER_GRAM'].describe()
            
            st.metric("Average Price (₹/g)", format_inr(stats['mean']))
            st.metric("Highest Price (₹/g)", format_inr(stats['max']))
            st.metric("Lowest Price (₹/g)", format_inr(stats['min']))
            st.metric("Price Change Level", format_inr(stats['std']))
            
        st.markdown("---")
        st.subheader("📋 Recent Data Table")
        st.dataframe(display_df.tail(10).style.format("{:,.2f}"), use_container_width=True)

    with tab2:
        st.subheader("🤖 Price Prediction (ARIMA)")
        st.markdown("Predicting future prices based on historical trends.")
        
        train_size = int(len(df) * 0.8)
        train_data = df['Price_PER_GRAM'].iloc[:train_size]

        if st.button("Start Prediction"):
            with st.spinner("Training model..."):
                try:
                    model_fit = train_arima_model(train_data, order=(5,1,0))
                    
                    future_forecast = model_fit.forecast(steps=forecast_days)
                    
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, inclusive='right')
                    
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price (₹/g)': future_forecast.values
                    }).set_index('Date')
                    
                    st.success("Prediction Complete!")
                    st.markdown(f"### 🔮 Forecast for Next {forecast_days} Days")
                    st.line_chart(future_df, color='#FFD700')
                    
                    pred_min = future_df['Predicted Price (₹/g)'].min()
                    pred_max = future_df['Predicted Price (₹/g)'].max()
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Minimum", format_inr(pred_min))
                    col2.metric("Predicted Maximum", format_inr(pred_max))
                    
                    st.session_state['forecast_df'] = future_df
                    
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab3:
        st.subheader("⬇️ Download Data")
        
        if 'forecast_df' in st.session_state:
            csv_forecast = st.session_state['forecast_df'].to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Prediction Data (CSV)",
                data=csv_forecast,
                file_name='gold_forecast_india.csv'
            )
        
        csv_hist = display_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Historical Data (CSV)",
            data=csv_hist,
            file_name='gold_historical_india.csv'
        )

# --- Main Execution Flow ---

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        show_login_ui()
    else:
        show_main_app()

if __name__ == "__main__":
    main()