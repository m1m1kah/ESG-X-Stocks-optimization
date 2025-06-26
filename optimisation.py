import numpy as np
import pandas as pd
import cvxpy as cp

def optimize_portfolio(predicted_returns, price_data, esg_data, monthly_returns, esg_weight=0.1, budget=5000):
    tickers = predicted_returns.index.tolist()
    n_assets = len(tickers)

    # Prices and ESG scores
    prices = price_data.loc[tickers, 'Latest Price']
    esg_scores = esg_data.loc[tickers, 'ESG_Score']

    # Use monthly historical returns to get covariance matrix
    returns_df = monthly_returns[tickers]
    cov_matrix = returns_df.cov() * 12  # Annualize

    expected_returns = predicted_returns

    # Optimization variables
    w = cp.Variable(n_assets)
    port_return = expected_returns.values @ w
    port_risk = cp.quad_form(w, cov_matrix.values)

    esg_threshold = esg_weight

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w @ esg_scores.values >= esg_threshold,
    ]

    risk_aversion = 0.1
    objective = cp.Maximize(port_return - risk_aversion * port_risk)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if w.value is None:
        raise ValueError("❌ Optimization failed. Try relaxing ESG threshold or check input data.")

    weights = pd.Series(w.value, index=tickers)

    portfolio_df = pd.DataFrame({
        "ticker": tickers,
        "weight": weights,
        "expected_return": expected_returns,
        "esg_score": esg_scores,
        "price": prices,
        "allocated": weights * budget
    }).round(4)

    portfolio_df = portfolio_df[portfolio_df["weight"] > 0.001]

    total_return = (expected_returns * weights).sum()
    avg_esg = (esg_scores * weights).sum()

    return {
        "portfolio": portfolio_df,
        "expected_return": total_return,
        "esg_score": avg_esg
    }

# === Load Data ===
monthly_returns = pd.read_csv('monthly_returns.csv', index_col=0, parse_dates=True)
latest_prices = pd.read_csv('latest_prices.csv', index_col=0)
esg_data = pd.read_csv('esg_percentile.csv', index_col=0)
forecast_returns = pd.read_csv('forecasted_results.csv', index_col=0)

# Rename column for consistency
forecast_returns.columns = ['Expected Return']
esg_data.columns = ['ESG_Score']

# Ensure consistent index formats
forecast_returns.index = forecast_returns.index.str.strip()
latest_prices.index = latest_prices.index.str.strip()
esg_data.index = esg_data.index.str.strip()
monthly_returns.columns = monthly_returns.columns.str.strip()

# Ensure consistent index formats
forecast_returns.index = forecast_returns.index.str.strip()
latest_prices.index = latest_prices.index.str.strip()
esg_data.index = esg_data.index.str.strip()
monthly_returns.columns = monthly_returns.columns.str.strip()

# === Only keep tickers that exist in all datasets ===
valid_tickers = set(forecast_returns.index) & set(latest_prices.index) & set(esg_data.index) & set(monthly_returns.columns)
valid_tickers = sorted(valid_tickers)


# Filter all dataframes
forecast_returns = forecast_returns.loc[valid_tickers]
latest_prices = latest_prices.loc[valid_tickers]
esg_data = esg_data.loc[valid_tickers]
monthly_returns = monthly_returns[valid_tickers]

# Now run optimizer
results = optimize_portfolio(
    predicted_returns=forecast_returns['Expected Return'],
    price_data=latest_prices,
    esg_data=esg_data,
    monthly_returns=monthly_returns,
    budget=5000
)


print(results['portfolio'])
print(f"Expected Annual Return: {results['expected_return']:.4f}")
print(f"Average ESG Score: {results['esg_score']:.4f}")
