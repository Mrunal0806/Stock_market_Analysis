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
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="📈 Stock Prediction App", layout="wide")
st.title("📊 STOCK MARKET PREDICTION USING PROPHET")

# -------------------------
# Sidebar Controls
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    tickers = st.multiselect(
        "Select Stock Tickers",
        ["AAPL", "MSFT", "TSLA", "RELIANCE.NS", "TCS.NS"],
        default=["AAPL"]
    )
    start_date = st.date_input("Start Date", datetime(2015, 1, 1))
    end_date = st.date_input("End Date", datetime.today() - timedelta(days=1))
    periods = st.slider("Forecast (in Days)", 30, 1826, 365)

# -------------------------
# Helper Functions
# -------------------------


@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[df["Volume"] > 0]  # Remove synthetic rows
    return df


def get_clean_close(df, ticker):
    try:
        if "Close" not in df.columns:
            st.warning(
                f"⚠️ 'Close' column missing for {ticker}. Columns: {df.columns.tolist()}")
            return None

        close_col = df["Close"]

        # If it's a DataFrame (due to duplicate columns), reduce to Series
        if isinstance(close_col, pd.DataFrame):
            if close_col.shape[1] == 1:
                close_col = close_col.iloc[:, 0]
            else:
                st.warning(
                    f"⚠️ 'Close' column for {ticker} is malformed. Shape: {close_col.shape}")
                return None

        if isinstance(close_col, pd.Series) and close_col.ndim == 1:
            if close_col.nunique() <= 1:
                st.warning(
                    f"⚠️ {ticker} data appears static or synthetic. Skipping.")
                return None
            return close_col

        st.warning(
            f"⚠️ Unexpected structure in 'Close' column for {ticker}. Type: {type(close_col)}")
        return None

    except Exception as e:
        st.warning(f"⚠️ Error validating 'Close' column for {ticker}: {e}")
        return None


def prepare_prophet_data(df):
    df_reset = df.reset_index()
    if "Date" not in df_reset.columns:
        df_reset.rename(columns={df_reset.columns[0]: "Date"}, inplace=True)
    df_prophet = df_reset[["Date", "Close"]].copy()
    df_prophet.columns = ["ds", "y"]
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
    df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")
    df_prophet.dropna(subset=["ds", "y"], inplace=True)
    return df_prophet


# -------------------------
# Forecasting Loop
# -------------------------
for ticker in tickers:
    with st.expander(f"📈 {ticker} Data from {start_date} to {end_date}", expanded=True):
        df = load_data(ticker, start_date, end_date)

        if df.empty:
            st.warning(f"⚠️ No data returned for {ticker}")
            continue

        close_series = get_clean_close(df, ticker)
        if close_series is None:
            continue

        st.write(
            f"✅ Close column for {ticker} → Series. Shape: {close_series.shape}")
        st.write("📋 Raw Data Preview:", df.tail())

        df_prophet = prepare_prophet_data(df)

        if df_prophet.empty:
            st.warning(
                f"❌ No clean data available for {ticker} after preprocessing.")
            continue

        # Train Prophet model
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)

        # Forecast future
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)

        # Evaluation (MAPE)
        try:
            actual = df_prophet["y"]
            predicted = forecast.loc[forecast["ds"].isin(
                df_prophet["ds"]), "yhat"]
            if not predicted.empty:
                mape = mean_absolute_percentage_error(actual, predicted)
                st.success(f"📈 MAPE for {ticker}: {mape:.2%}")
        except Exception as e:
            st.warning(f"⚠️ Error calculating MAPE: {e}")

        # Bollinger Bands
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
        # Forecast Components
        # -------------------------
        st.subheader("📊 Forecast Decomposition")
        fig_components = plot_components_plotly(m, forecast)
        st.plotly_chart(fig_components, use_container_width=True)

        # -------------------------
        # Candlestick + Indicators
        # -------------------------
        st.subheader("📊 Technical Indicators")

        df["SMA_20"] = df["Close"].rolling(window=20).mean()

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["SMA_20"],
            mode="lines",
            name="SMA 20",
            line=dict(color="orange", dash="dash")
        ))

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

        fig.add_trace(go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(0,0,0,0.2)",
            yaxis="y2",
            opacity=0.4
        ))

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
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # Download Forecast CSV
        # -------------------------
        st.download_button(
            label="📥 Download Forecast CSV",
            data=forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )
