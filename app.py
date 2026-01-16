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
st.set_page_config(page_title="Pro AI Stock Terminal", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡 Pro AI Stock Terminal (Ultimate Edition)")

# 2. Sidebar Search Logic
st.sidebar.title("🔍 Search Stock")
common_stocks = ["Select...", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ZOMATO.NS", "TATAMOTORS.NS", "SBIN.NS", "AAPL", "NVDA"]
selected_from_list = st.sidebar.selectbox("Quick Selection:", options=common_stocks)
custom_ticker = st.sidebar.text_input("OR Type Name (e.g. MRF, TITAN):").upper()

# Intelligent Ticker Logic
if custom_ticker:
    selected_stock = custom_ticker + ".NS" if "." not in custom_ticker else custom_ticker
elif selected_from_list != "Select...":
    selected_stock = selected_from_list
else:
    selected_stock = "RELIANCE.NS"

n_years = st.sidebar.slider("Prediction Period (Years):", 1, 5)

# 3. Data Loading (Stability Fix)
@st.cache_data(ttl=3600) # १ तास डेटा कॅश राहील
def load_data(ticker):
    try:
        # auto_adjust आणि threads=False मुळे Cloud वर स्थिरता वाढते
        data = yf.download(ticker, start="2015-01-01", auto_adjust=True, threads=False)
        if data.empty: return None
        data.reset_index(inplace=True)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        return data
    except: return None

data = load_data(selected_stock)

if data is None:
    st.error(f"❌ '{selected_stock}' सापडला नाही! कृपया स्पेलिंग तपासा.")
else:
    # 4. Financial Metrics (SAFE FETCHING to avoid N/A)
    curr_price = float(data['Close'].iloc[-1])
    ticker_obj = yf.Ticker(selected_stock)
    
    # Rate Limit एरर टाळण्यासाठी सुरक्षित पद्धत
    try:
        fast_info = ticker_obj.fast_info
        m_cap = fast_info.get('market_cap', 'N/A')
    except:
        m_cap = 'N/A'

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"{curr_price:.2f}")
    
    # Market Cap Display Logic
    if isinstance(m_cap, (int, float)):
        col2.metric("Market Cap", f"{m_cap:,.0f}")
    else:
        col2.metric("Market Cap", "N/A (Limit)")
        
    col3.metric("52W High", f"{data['Close'].max():.2f}")
    col4.metric("52W Low", f"{data['Close'].min():.2f}")

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
        if last_rsi > 70: st.error(f"🎯 SIGNAL: SELL (RSI: {last_rsi:.2f})")
        elif last_rsi < 30: st.success(f"🎯 SIGNAL: BUY (RSI: {last_rsi:.2f})")
        else: st.info(f"🎯 SIGNAL: HOLD (RSI: {last_rsi:.2f})")
    with c2:
        volatility = data['Close'].tail(30).pct_change().std() * 100
        st.warning(f"⚠ Risk Assessment: {'HIGH' if volatility > 1.5 else 'LOW'} ({volatility:.2f}%)")

    # 6. Advanced Chart: Price + EMA + Volume
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Price", line=dict(color='#00d1ff')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA50'], name="EMA 50", line=dict(color='yellow')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA200'], name="EMA 200", line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="Volume", marker_color='gray', opacity=0.4), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 7. AI Prediction
    st.divider()
    st.subheader(f"🚀 AI {n_years}-Year Forecast")
    df_train = pd.DataFrame({'ds': pd.to_datetime(data['Date']).dt.tz_localize(None), 'y': data['Close']}).dropna()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=n_years * 365)
    forecast = m.predict(future)
    st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)

    # 8. News Sentiment
    st.divider()
    st.subheader("📰 Market Sentiment")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={selected_stock}+stock&hl=en-IN")
    for entry in feed.entries[:3]:
        analysis = TextBlob(entry.title)
        icon = "🟢" if analysis.sentiment.polarity > 0 else ("🔴" if analysis.sentiment.polarity < 0 else "⚪")
        st.write(f"{icon} {entry.title}")
