import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import feedparser
from textblob import TextBlob

# 1. Page Configuration
st.set_page_config(page_title="Universal AI Stock Terminal V2", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡 Pro AI Stock Terminal (Ultimate Edition)")

# 2. Sidebar Search Logic
st.sidebar.title("🔍 Search Stock")
# अधिक भारतीय स्टॉक्सची यादी जोडली आहे
common_stocks = [
    "Select...", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ZOMATO.NS", 
    "TATAMOTORS.NS", "SBIN.NS", "INFY.NS", "BAJFINANCE.NS", "AAPL", "TSLA"
]
selected_from_list = st.sidebar.selectbox("Quick Selection:", options=common_stocks)
custom_ticker = st.sidebar.text_input("OR Type Ticker (e.g. MRF, TITAN):").upper()

# Intelligent Ticker Logic
if custom_ticker:
    # जर युजरने .NS लावले नसेल तर भारतीय मार्केटसाठी ते स्वयंचलितपणे जोडले जाईल
    if "." not in custom_ticker:
        selected_stock = custom_ticker + ".NS"
    else:
        selected_stock = custom_ticker
elif selected_from_list != "Select...":
    selected_stock = selected_from_list
else:
    selected_stock = "RELIANCE.NS"

n_years = st.sidebar.slider("Prediction Period (Years):", 1, 5)

# 3. Enhanced Data Loading (With Stability Fix)
@st.cache_data
def load_data(ticker):
    try:
        # threads=False आणि auto_adjust=True मुळे भारतीय डेटा लोड होण्यास मदत होते
        data = yf.download(ticker, start="2015-01-01", auto_adjust=True, threads=False)
        if data.empty: return None
        data.reset_index(inplace=True)
        # MultiIndex कॉलम्स फिक्स करणे
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        return None

data = load_data(selected_stock)

if data is None:
    st.error(f"❌ '{selected_stock}' सापडला नाही! कृपया Yahoo Finance वरील सिम्बॉल वापरा (उदा. TCS.NS, MRF.NS).")
else:
    # 4. Financial Metrics
    ticker_obj = yf.Ticker(selected_stock)
    info = ticker_obj.info
    curr_price = float(data['Close'].iloc[-1])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"{curr_price:.2f}")
    col2.metric("Market Cap", f"{info.get('marketCap', 'N/A'):,.0f}")
    col3.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
    col4.metric("Debt-to-Equity", info.get('debtToEquity', 'N/A'))

    # 5. Technical Signals
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    last_rsi = 100 - (100 / (1 + rs.iloc[-1]))

    st.divider()
    st.subheader("🛠 AI Technical Analysis")
    c1, c2 = st.columns(2)
    with c1:
        if last_rsi > 70:
            st.error(f"🎯 SIGNAL: SELL (RSI: {last_rsi:.2f} - Overbought)")
        elif last_rsi < 30:
            st.success(f"🎯 SIGNAL: BUY (RSI: {last_rsi:.2f} - Oversold)")
        else:
            st.info(f"🎯 SIGNAL: HOLD (RSI: {last_rsi:.2f} - Neutral)")
    with c2:
        volatility = data['Close'].tail(30).pct_change().std() * 100
        risk = "LOW" if volatility < 1.5 else "HIGH"
        st.warning(f"⚠ Risk Assessment: {risk} ({volatility:.2f}%)")

    # 6. Technical Chart (Price + EMA + Volume)
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

    st.divider()
    st.subheader("📈 Pro Technical Chart")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Price", line=dict(color='#00d1ff')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA50'], name="EMA 50", line=dict(color='yellow', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA200'], name="EMA 200", line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="Volume", marker_color='gray', opacity=0.4), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 7. AI Prediction
    st.divider()
    st.subheader(f"🚀 AI {n_years}-Year Forecast (Smart Seasonality)")
    df_train = pd.DataFrame({'ds': pd.to_datetime(data['Date']).dt.tz_localize(None), 'y': data['Close']}).dropna()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=n_years * 365)
    forecast = m.predict(future)
    st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)

    # Verdict Box
    forecast_val = float(forecast['yhat'].iloc[-1])
    change_pct = ((forecast_val - curr_price) / curr_price) * 100
    st.write("### 🎯 AI Forecast Verdict")
    if change_pct > 0:
        st.success(f"AI predicts a potential *{change_pct:.2f}% upside* in {n_years} years. (Target: {forecast_val:.2f})")
    else:
        st.warning(f"AI predicts a potential *{abs(change_pct):.2f}% downside* in {n_years} years. (Support: {forecast_val:.2f})")

    # 8. Sentiment Analysis
    st.divider()
    st.subheader("📰 Market Sentiment Analysis")
    rss_url = f"https://news.google.com/rss/search?q={selected_stock}+stock&hl=en-IN"
    feed = feedparser.parse(rss_url)
    sent_scores = []
    for entry in feed.entries[:5]:
        analysis = TextBlob(entry.title)
        sentiment = "Positive" if analysis.sentiment.polarity > 0 else ("Negative" if analysis.sentiment.polarity < 0 else "Neutral")
        icon = "🟢" if sentiment == "Positive" else ("🔴" if sentiment == "Negative" else "⚪")
        st.write(f"{icon} {entry.title}")
        sent_scores.append(analysis.sentiment.polarity)
    
    if sent_scores:
        avg_sent = sum(sent_scores)/len(sent_scores)
        st.info(f"Overall Market Sentiment Score: *{avg_sent:.2f}* (-1 to 1)")
