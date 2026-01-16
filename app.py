import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from textblob import TextBlob
import feedparser

# 1. UI Configuration
st.set_page_config(page_title="Ultimate AI Stock Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, span, label {
        color: #121212 !important; 
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff; border: 1px solid #edeff5;
        padding: 15px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    [data-testid="stMetricValue"] { color: #121212 !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡 Universal AI Stock Terminal")

# 2. Sidebar: Settings & New Interval Option
st.sidebar.markdown("### 🔍 Search & Options")
common_stocks = ["Select...", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ZOMATO.NS", "TATAMOTORS.NS", "AAPL", "NVDA", "TSLA"]
selected_from_list = st.sidebar.selectbox("Search Recommendations:", options=common_stocks)
custom_ticker = st.sidebar.text_input("OR Type Stock Name:").upper()

if custom_ticker:
    selected_stock = custom_ticker + ".NS" if "." not in custom_ticker else custom_ticker
elif selected_from_list != "Select...":
    selected_stock = selected_from_list
else:
    selected_stock = "RELIANCE.NS"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Prediction Settings")
# NAVIN OPTION: Prediction Interval (5m pasun pudhe)
time_interval = st.sidebar.selectbox("Select Prediction Interval:", 
                                   options=["5m", "15m", "1h", "1d"], index=3)
n_periods = st.sidebar.number_input("Forecast Duration (Steps):", min_value=1, value=30)

# 3. Data Loading (Flexible for Intervals)
@st.cache_data(ttl=60)
def load_dynamic_data(ticker, interval):
    try:
        # Interval pramane period set karne
        period_map = {"5m": "1d", "15m": "5d", "1h": "1mo", "1d": "max"}
        data = yf.download(ticker, period=period_map[interval], interval=interval, auto_adjust=True, threads=False)
        if data.empty: return None
        data.reset_index(inplace=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except: return None

# --- या ओळीच्या जागी (Step 3 शोधून) ---
with st.spinner(f"Fetching latest data for {selected_stock}..."):
    data = load_dynamic_data(selected_stock, time_interval)
# -----------------------------------

if data is not None:
    # 4. Key Metrics
    curr_price = float(data['Close'].iloc[-1])
    prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else curr_price
    change = curr_price - prev_close
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"₹{curr_price:.2f}", f"{change:.2f} ({(change/prev_close)*100:.2f}%)")
    
    ticker_obj = yf.Ticker(selected_stock)
    try:
        f_info = ticker_obj.fast_info
        m_cap = f_info.get('market_cap', 0) / 10**7
        col2.metric("Market Cap", f"₹{m_cap:,.0f} Cr")
    except: col2.metric("Market Cap", "N/A")
    
    col3.metric("High", f"₹{float(data['High'].max()):.2f}")
    col4.metric("Low", f"₹{float(data['Low'].min()):.2f}")

    # 5. Dynamic Prediction Chart (Fix for Intraday)
    st.divider()
    st.subheader(f"🚀 AI Forecast ({time_interval} Interval)")
    try:
        # Time column handle karne (Datetime ki Date)
        time_col = data.columns[0]
        df_p = pd.DataFrame({
            'ds': pd.to_datetime(data[time_col]).dt.tz_localize(None),
            'y': data['Close'].values.flatten()
        })
        
        m = Prophet(changepoint_prior_scale=0.05).fit(df_p)
        
        # Interval nusar frequency set karne
        freq_map = {"5m": "5min", "15m": "15min", "1h": "H", "1d": "D"}
        future = m.make_future_dataframe(periods=int(n_periods), freq=freq_map[time_interval])
        forecast = m.predict(future)
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', 
                                     line=dict(color='#00d09c', width=3), name='AI Prediction'))
        fig_pred.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], mode='lines', 
                                     line=dict(color='#121212', width=1), name='Actual Price'))
        fig_pred.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=500, xaxis_title="Time")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        target = forecast['yhat'].iloc[-1]
        st.write(f"### 🎯 AI Verdict: Expected Price after {n_periods} steps: ₹{target:.2f}")

        # --- नवीन ऍड केलेला विभाग: AI Analysis Summary ---
        st.info("📊 *AI Analysis Summary (अंदाज कशावर आधारित आहे?)*")
        recent_trend = "Upward (वाढता)" if target > curr_price else "Downward (घसरता)"
        
        # साधे तांत्रिक विश्लेषण
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        ma_status = "Positive" if curr_price > sma_20 else "Negative"

        st.write(f"""
        * *Trend Analysis:* AI ने मागील डेटा पॅटर्नचा अभ्यास करून हा *{recent_trend}* कल वर्तवला आहे.
        * *Technical Indicator:* सध्या शेअरची किंमत २०-पीरियड सरासरीच्या (SMA) *{'वर' if ma_status == 'Positive' else 'खाली'}* आहे, जे {ma_status} संकेत दर्शवते.
        * *Forecast:* पुढील {n_periods} स्टेप्समध्ये किंमत साधारण *₹{target:.2f}* पर्यंत जाण्याची शक्यता आहे.
        """)
        # -----------------------------------------------

    except Exception as e:
        st.warning(f"Prediction Error: {e}")

    # 6. Candlestick Chart
    st.divider()
    st.subheader(f"Performance Chart ({time_interval})")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=data[data.columns[0]], open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Bar(x=data[data.columns[0]], y=data['Volume'], name="Volume", marker_color='#e0e3eb'), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", height=500, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

    # 7. Intraday Analysis (VWAP)
    if time_interval in ["5m", "15m"]:
        st.divider()
        st.subheader("⚡ Intraday Day Trading Signal")
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        last_p, last_v = data['Close'].iloc[-1], data['VWAP'].iloc[-1]
        if last_p > last_v: st.success(f"BULLISH: Price is above VWAP (₹{last_v:.2f})")
        else: st.error(f"BEARISH: Price is below VWAP (₹{last_v:.2f})")

    # 8. News
    st.divider()
    st.subheader("📰 Latest Market News")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={selected_stock}+stock&hl=en-IN")
    for entry in feed.entries[:5]:
        st.write(f"⚪ {entry.title}")

else:
    st.error("Data not found. Please check your selection.")
