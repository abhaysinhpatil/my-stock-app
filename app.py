import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from textblob import TextBlob
import pandas as pd
import feedparser # ताज्या बातम्यांसाठी

# १. सेटिंग्ज
st.set_page_config(page_title="Pro AI Stock Analyst", layout="wide")
st.title("📊 Pro AI Stock Analyst & News Tracker")

# २. साईडबार - स्टॉक आणि इंडिकेटर्स निवडणे
stocks = ("AAPL", "GOOG", "MSFT", "TSLA", "RELIANCE.NS", "TATASTEEL.NS")
selected_stock = st.sidebar.selectbox("Select Stock", stocks)
n_years = st.sidebar.slider("Prediction Years:", 1, 5)

# ३. डेटा लोड करणे
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2015-01-01")
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

data = load_data(selected_stock)

# ४. Technical Indicators (Moving Averages)
# २० दिवसांची आणि ५० दिवसांची सरासरी काढणे
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# ५. Live News Section (Google News RSS)
st.subheader(f"📰 Live News & Sentiment: {selected_stock}")
rss_url = f"https://news.google.com/rss/search?q={selected_stock}+stock&hl=en-IN&gl=IN&ceid=IN:en"
feed = feedparser.parse(rss_url)

col1, col2 = st.columns([2, 1])

with col2:
    st.write("Latest Headlines:")
    for entry in feed.entries[:5]: # पहिल्या ५ बातम्या
        sentiment = TextBlob(entry.title).sentiment.polarity
        icon = "✅" if sentiment > 0 else "❌" if sentiment < 0 else "⚪"
        st.write(f"{icon} [{entry.title}]({entry.link})")

# ६. ऐतिहासिक ग्राफ + Technical Indicators
with col1:
    fig = go.Figure()
    clean_date = pd.to_datetime(data['Date']).dt.tz_localize(None)
    fig.add_trace(go.Scatter(x=clean_date, y=data['Close'], name="Close Price", line=dict(color='white')))
    fig.add_trace(go.Scatter(x=clean_date, y=data['MA20'], name="20 Day MA", line=dict(color='cyan', dash='dot')))
    fig.add_trace(go.Scatter(x=clean_date, y=data['MA50'], name="50 Day MA", line=dict(color='magenta', dash='dot')))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ७. AI Prediction (Prophet) - रेषा दाखवण्यासाठी सुधारित
st.subheader('🚀 AI Price Forecast (Line View)')

# मॉडेल ट्रेनिंग आणि प्रेडिक्शन
period = n_years * 365
df_train = pd.DataFrame({'ds': pd.to_datetime(data['Date']).dt.tz_localize(None), 'y': data['Close']}).dropna()

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# नवीन ग्राफ तयार करणे (डॉट्स काढून रेषा वापरण्यासाठी)
fig_forecast = go.Figure()

# १. खरा जुना डेटा (Actual Data) - आता रेषेच्या स्वरूपात
fig_forecast.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name="Actual Price", line=dict(color='white', width=1)))

# २. प्रेडिक्शन (Forecast) - मध्यवर्ती रेषा
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Trend", line=dict(color='#00d1ff', width=2)))

# ३. सावली (Confidence Interval) - अनिश्चितता दर्शवण्यासाठी
fig_forecast.add_trace(go.Scatter(
    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(0, 209, 255, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False,
    name='Uncertainty'
))

fig_forecast.update_layout(template="plotly_dark", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_forecast, use_container_width=True)

# ८. Risk Meter (Volatility Analysis)
st.subheader("⚠ Risk Assessment (Volatility)")

# गेल्या ३० दिवसांच्या बदलावरून जोखीम मोजणे
recent_data = data['Close'].tail(30)
volatility = recent_data.pct_change().std() * 100 # Standard Deviation

col_risk1, col_risk2 = st.columns(2)

with col_risk1:
    if volatility < 1.5:
        st.success(f"Low Risk (Volatility: {volatility:.2f}%)")
        st.write("हा स्टॉक सध्या स्थिर आहे आणि यात मोठी घसरण होण्याची शक्यता कमी दिसते.")
    elif 1.5 <= volatility < 2.5:
        st.warning(f"Medium Risk (Volatility: {volatility:.2f}%)")
        st.write("यात मध्यम स्वरूपाची अस्थिरता आहे. गुंतवणूक करताना सावध राहा.")
    else:
        st.error(f"High Risk (Volatility: {volatility:.2f}%)")
        st.write("हा स्टॉक अत्यंत अस्थिर आहे. यात पैसे गुंतवणे जोखमीचे ठरू शकते.")

with col_risk2:
    # एक साधा प्रोग्रेस बार जो मीटरसारखा दिसेल
    st.write("Risk Level Visualization:")
    risk_score = min(volatility * 20, 100) # Score out of 100
    st.progress(int(risk_score))

    # आधीचा कोड संपल्यानंतर इथे खाली पेस्ट करा...

# --- न्यूज आधारित प्रेडिक्शन विभाग ---
st.divider() # एक रेषा ओढण्यासाठी
st.subheader(f"🧠 AI News-Based Analysis for {selected_stock}")

def get_news_prediction(ticker):
    # गुगल न्यूजवरून बातम्या शोधणे
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    
    total_score = 0
    count = 0
    
    # पहिल्या १० बातम्यांचे विश्लेषण करणे
    for entry in feed.entries[:10]:
        analysis = TextBlob(entry.title).sentiment.polarity
        total_score += analysis
        count += 1
    
    avg_score = total_score / count if count > 0 else 0
    return avg_score

news_score = get_news_prediction(selected_stock)

# रिझल्ट दाखवणे
if news_score > 0.1:
    st.success(f"🚀 POSITIVE TREND: बातम्यांनुसार या स्टॉकमध्ये वाढ होण्याची शक्यता आहे. (Sentiment Score: {news_score:.2f})")
elif news_score < -0.1:
    st.error(f"⚠ NEGATIVE TREND: बातम्या सध्या नकारात्मक आहेत, गुंतवणूक करताना काळजी घ्या. (Sentiment Score: {news_score:.2f})")
else:
    st.info(f"⚖ NEUTRAL: बातम्यांमध्ये कोणताही मोठा बदल दिसत नाही. (Sentiment Score: {news_score:.2f})")