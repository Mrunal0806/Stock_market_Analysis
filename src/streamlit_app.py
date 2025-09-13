# import streamlit as st
# import yfinance as yf
# import pandas as pd
# from prophet import Prophet
# from prophet.plot import plot_plotly
# from datetime import datetime

# # -------------------------
# # Page Setup
# # -------------------------
# st.set_page_config(page_title="📈 Stock Prediction App", layout="wide")
# st.title("📊 STOCK MARKET PREDICTION USING PROPHET")

# # -------------------------
# # User Inputs
# # -------------------------
# tickers = st.multiselect(
#     "Select Stock Tickers (e.g. AAPL, MSFT, TSLA, RELIANCE.NS)",
#     ["AAPL", "MSFT", "TSLA", "RELIANCE.NS"],
#     default=["AAPL"]
# )

# start_date = st.date_input("Start Date", datetime(2015, 1, 1))
# end_date = st.date_input("End Date", datetime.today())
# periods = st.slider("Forecast Days into Future", 30, 365, 180)

# # -------------------------
# # Data Loader
# # -------------------------
# @st.cache_data
# def load_data(ticker, start, end):
#     return yf.download(ticker, start=start, end=end)

# # -------------------------
# # Forecasting Loop
# # -------------------------
# for ticker in tickers:
#     st.subheader(f"📌 {ticker} Data from {start_date} to {end_date}")
#     df = load_data(ticker, start_date, end_date)

#     if df.empty or "Close" not in df.columns or "Date" not in df.reset_index().columns:
#         st.warning(f"⚠️ No valid 'Close' or 'Date' column found for {ticker}")
#         continue

#     st.write(df.tail())

#     # Prepare data for Prophet
#     df_prophet = df.reset_index()[["Date", "Close"]].copy()
#     df_prophet.columns = ["ds", "y"]  # Rename to Prophet format

#     # Ensure correct types
#     df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
#     df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")

#     # Drop rows with missing values
#     if not all(col in df_prophet.columns for col in ["ds", "y"]):
#         st.warning(f"❌ Missing required columns for Prophet in {ticker}")
#         continue

#     df_prophet.dropna(subset=["ds", "y"], inplace=True)

#     if df_prophet.empty:
#         st.warning(f"❌ No clean data available for {ticker} after preprocessing.")
#         continue

#     # Train Prophet model
#     m = Prophet(daily_seasonality=True)
#     m.fit(df_prophet)

#     # Forecast future
#     future = m.make_future_dataframe(periods=periods)
#     forecast = m.predict(future)

#     # Plot forecast
#     st.subheader(f"🔮 Prophet Forecast - {ticker}")
#     fig = plot_plotly(m, forecast)
#     st.plotly_chart(fig, use_container_width=True)

#     # Forecast components
#     st.subheader("📊 Forecast Components")
#     st.write(m.plot_components(forecast))

import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime
import plotly.graph_objects as go

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="📈 Stock Prediction App", layout="wide")
st.title("📊 STOCK MARKET PREDICTION USING PROPHET")

# -------------------------
# User Inputs
# -------------------------
tickers = st.multiselect(
    "Select Stock Tickers (e.g. AAPL, MSFT, TSLA, RELIANCE.NS)",
    ["AAPL", "MSFT", "TSLA", "RELIANCE.NS"],
    default=["AAPL"]
)

start_date = st.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.date_input("End Date", datetime.today())
periods = st.slider("Forecast Days into Future", 30, 365, 180)

# -------------------------
# Data Loader
# -------------------------
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# -------------------------
# Forecasting Loop
# -------------------------
for ticker in tickers:
    st.subheader(f"📌 {ticker} Data from {start_date} to {end_date}")
    df = load_data(ticker, start_date, end_date)

    if df.empty or "Close" not in df.columns or "Date" not in df.reset_index().columns:
        st.warning(f"⚠️ No valid 'Close' or 'Date' column found for {ticker}")
        continue

    st.write(df.tail())

    # Prepare data for Prophet
    df_prophet = df.reset_index()[["Date", "Close"]].copy()
    df_prophet.columns = ["ds", "y"]
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
    df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")
    df_prophet.dropna(subset=["ds", "y"], inplace=True)

    if df_prophet.empty:
        st.warning(f"❌ No clean data available for {ticker} after preprocessing.")
        continue

    # Train Prophet model
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)

    # Forecast future
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    # Add Bollinger Bands to forecast
    forecast["ma20"] = forecast["yhat"].rolling(window=20).mean()
    forecast["std20"] = forecast["yhat"].rolling(window=20).std()
    forecast["bollinger_upper"] = forecast["ma20"] + 2 * forecast["std20"]
    forecast["bollinger_lower"] = forecast["ma20"] - 2 * forecast["std20"]

    # -------------------------
    # Prophet Forecast Plot
    # -------------------------
    st.subheader(f"🔮 Prophet Forecast - {ticker}")
    fig_forecast = plot_plotly(m, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # -------------------------
    # Custom Forecast Components with Candlestick
    # -------------------------
    st.subheader("📊 Forecast Components with Technical Indicators")

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    # Prophet trend line
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["trend"],
        mode="lines",
        name="Prophet Trend",
        line=dict(color="blue", width=2)
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["bollinger_upper"],
        mode="lines",
        name="Upper Band",
        line=dict(color="green", dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["bollinger_lower"],
        mode="lines",
        name="Lower Band",
        line=dict(color="red", dash="dot")
    ))

    # Volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color="rgba(0,0,0,0.2)",
        yaxis="y2",
        opacity=0.4
    ))

    # Layout adjustments
    fig.update_layout(
        title=f"{ticker} Forecast Components",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)