import streamlit as st
import pandas as pd
import numpy as np
from optimisation import optimize_portfolio

# === Load full data once ===
@st.cache_data
def load_data():
    forecast = pd.read_csv('forecasted_results.csv', index_col=0)
    forecast.columns = ['Expected Return']
    esg = pd.read_csv('esg_percentile.csv', index_col=0)
    esg.columns = ['ESG_Score']
    prices = pd.read_csv('latest_prices.csv', index_col=0)
    returns = pd.read_csv('monthly_returns.csv', index_col=0, parse_dates=True)

    # Clean tickers
    for df in [forecast, esg, prices]:
        df.index = df.index.str.strip()
    returns.columns = returns.columns.str.strip()

    valid = list(set(forecast.index) & set(esg.index) & set(prices.index) & set(returns.columns))
    valid = sorted(valid)

    return (
        forecast.loc[valid], 
        prices.loc[valid], 
        esg.loc[valid], 
        returns[valid]
    )

forecast_returns, latest_prices, esg_data, monthly_returns = load_data()

# === UI ===
st.title("🌍 ESG-Aware Portfolio Optimizer")

tickers_input = st.text_input("Enter tickers (comma-separated)", ", ".join(forecast_returns.index[:6]))
investment_amount = st.number_input("💰 Total Investment Amount (£)", min_value=1000, max_value=1_000_000, value=5000, step=500)
esg_weight = st.slider("Minimum ESG Score", 0.0, 1.0, 0.7, step=0.05)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
missing = [t for t in tickers if t not in forecast_returns.index]

if missing:
    st.warning(f"⚠️ These tickers are missing from your data: {', '.join(missing)}")
elif st.button("🔍 Optimize Portfolio") and len(tickers) > 1:
    st.subheader("📈 Optimizing...")

    result = optimize_portfolio(
        predicted_returns=forecast_returns.loc[tickers]['Expected Return'],
        price_data=latest_prices.loc[tickers],
        esg_data=esg_data.loc[tickers],
        monthly_returns=monthly_returns[tickers],
        esg_weight=esg_weight,
        budget=investment_amount
    )

    st.markdown(f"💼 Total Budget: **£{investment_amount:,}**")

    st.subheader("📊 Optimized Portfolio")
    st.dataframe(result["portfolio"].set_index("ticker").style.format({
        "weight": "{:.2%}",
        "expected_return": "{:.2%}",
        "esg_score": "{:.2f}",
        "price": "£{:.2f}",
        "allocated": "£{:.2f}"
    }), use_container_width=True)

    st.markdown(f"""
    - **Expected Portfolio Return**: `{result['expected_return']:.2%}`
    - **Average ESG Score**: `{result['esg_score']:.2f}`
    """)
