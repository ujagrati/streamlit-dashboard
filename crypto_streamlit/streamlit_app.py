import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.seasonal import STL

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_crypto.csv", parse_dates=["Date"])

df = load_data()

st.title("📊 Cryptocurrency Market Dashboard")

# Sidebar - Coin Selection
coins = df['Coin'].unique()
selected_coin = st.sidebar.selectbox("Select a Coin", coins)
coin_df = df[df['Coin'] == selected_coin].sort_values('Date').copy()

# Current Price
latest = coin_df.iloc[-1]
st.metric("Current Price (USD)", f"${latest['Close']:.2f}")

# Main Panel - Price and Market Capitalization
st.subheader(f"📈 {selected_coin} Price & Market Capitalization")
st.markdown("""
**How it's calculated:**  
- *Price:* Taken from the closing price of the cryptocurrency.
- *Market Cap:* Indicates total value = price × circulating supply.

**What we learn:**  
Track how investor confidence and market value evolve over time.
""")
fig_price = px.line(coin_df, x='Date', y='Close', title=f"{selected_coin} Closing Price Over Time",
                    labels={'Close': 'Closing Price (USD)', 'Date': 'Date'})
st.plotly_chart(fig_price)

fig_marketcap = px.line(coin_df, x='Date', y='Marketcap', title=f"{selected_coin} Market Capitalization Over Time",
                        labels={'Marketcap': 'Market Capitalization (USD)', 'Date': 'Date'})
st.plotly_chart(fig_marketcap)


# Price Forecast using Prophet
st.subheader(f"🔮 {selected_coin} - 30-Day Price Forecast")
st.markdown("""
**How it's calculated:**  
- Correlation of daily returns between the selected coin and others.

**What we learn:**  
- High correlation: moves with the market.
- Low or negative correlation: more independent — useful for portfolio diversification.
""")
st.markdown("""
This forecast is generated using Facebook's Prophet model. It predicts the **closing price** for the next 30 days,
including upper and lower bounds indicating the **confidence interval** of the forecast.
""")

# Prepare data for Prophet
df_prophet = coin_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot Forecast
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Price', line=dict(color='blue')))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='lightblue')))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='lightblue'), fill='tonexty', fillcolor='rgba(173,216,230,0.2)'))
fig_forecast.update_layout(title=f"{selected_coin} Forecasted Price with Confidence Interval",
                           xaxis_title="Date", yaxis_title="Forecasted Closing Price (USD)")
st.plotly_chart(fig_forecast)

# Volatility Comparison
st.sidebar.subheader("📊 Volatility Analysis")
volatility_df = df.groupby('Coin')['Return'].std().reset_index().rename(columns={'Return': 'Volatility'})
st.sidebar.write("Most Stable Coins:")
st.sidebar.dataframe(volatility_df.sort_values('Volatility').head(5))
st.sidebar.write("Most Volatile Coins:")
st.sidebar.dataframe(volatility_df.sort_values('Volatility', ascending=False).head(5))

# Buyer Recommendation Section
st.sidebar.subheader("💡 Buyer Recommendation")
recommended = volatility_df.sort_values('Volatility').head(1)['Coin'].values[0]
st.sidebar.success(f"Based on volatility and stability, consider investing in: **{recommended}**")

# Correlation heatmap (Selected Coin vs Others)
st.subheader(f"{selected_coin} Correlation with Other Coins")
st.markdown("""
**How it's calculated:**  
- Correlation of daily returns between the selected coin and others.

**What we learn:**  
- High correlation: moves with the market.
- Low or negative correlation: more independent — useful for portfolio diversification.
""")

# Sidebar toggle
drop_na = st.sidebar.checkbox("Only include dates with full data", value=True)

# Pivot return data
pivot_df = df.pivot_table(index='Date', columns='Coin', values='Return')

if drop_na:
    pivot_df_clean = pivot_df.dropna()
else:
    pivot_df_clean = pivot_df.copy()

# Compute correlation matrix
corr_matrix = pivot_df_clean.corr()

# Extract selected coin's correlation with others
if selected_coin in corr_matrix.columns:
    selected_corr = corr_matrix[[selected_coin]].dropna().sort_values(by=selected_coin, ascending=False)

    fig_selected_corr = px.bar(
        selected_corr,
        y=selected_corr.index,
        x=selected_coin,
        orientation='h',
        title=f"Correlation of {selected_coin} with Other Coins",
        labels={selected_coin: "Correlation Coefficient", "index": "Coin"},
        color=selected_coin,
        color_continuous_scale="Blues",
        range_x=[-1, 1]
    )
    st.plotly_chart(fig_selected_corr)
else:
    st.warning(f"No correlation data available for {selected_coin}.")


# Seasonal Trend Decomposition
st.subheader(f"🔁 {selected_coin} Seasonal Trend Decomposition")
st.markdown("""
**How it's calculated:**  
- STL decomposition splits price into **Trend**, **Seasonal**, and **Noise**.

**What we learn:**  
- Detects recurring patterns (e.g., monthly cycles).
- Helps separate market rhythm from randomness.
""")
coin_df.set_index('Date', inplace=True)
stl = STL(coin_df['Close'], period=30)
res = stl.fit()
fig_seasonal = px.line(x=res.seasonal.index, y=res.seasonal.values,
                       title="Seasonal Component of Price Fluctuation",
                       labels={'x': 'Date', 'y': 'Seasonal Variation'})
st.plotly_chart(fig_seasonal)
