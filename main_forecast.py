import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load data
df_long = pd.read_csv('sp500_close_long.csv', parse_dates=['Date'])

# Quick check
print(df_long.head())
print(df_long['Ticker'].unique()[:5])  # some tickers

# Step 2: Feature engineering

# Sort data by ticker and date
df_long = df_long.sort_values(['Ticker', 'Date'])

# Calculate daily returns
df_long['Return'] = df_long.groupby('Ticker')['Close'].pct_change()

# Rolling features - rolling mean and std dev of returns over 5 and 20 days
df_long['Return_MA_5'] = df_long.groupby('Ticker')['Return'].transform(lambda x: x.rolling(5).mean())
df_long['Return_STD_5'] = df_long.groupby('Ticker')['Return'].transform(lambda x: x.rolling(5).std())
df_long['Return_MA_20'] = df_long.groupby('Ticker')['Return'].transform(lambda x: x.rolling(20).mean())
df_long['Return_STD_20'] = df_long.groupby('Ticker')['Return'].transform(lambda x: x.rolling(20).std())

# Drop rows with NA values from rolling
df_long = df_long.dropna()

# Step 3: Prepare target - 1 year forward return
# Assuming roughly 252 trading days in a year
df_long['Return_1Y'] = df_long.groupby('Ticker')['Close'].transform(lambda x: x.shift(-252)/x - 1)

# Drop last 252 rows per ticker which won't have target
df_long = df_long.dropna(subset=['Return_1Y'])


# Step 4: Model training per ticker

results = []

tickers = df_long['Ticker'].unique()

for ticker in tickers:
    df_ticker = df_long[df_long['Ticker'] == ticker]
    
    features = ['Return', 'Return_MA_5', 'Return_STD_5', 'Return_MA_20', 'Return_STD_20']
    X = df_ticker[features]
    y = df_ticker['Return_1Y']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': 1,
        'boosting_type': 'gbdt',
        'seed': 42
    }
    
    model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)
    
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f'{ticker}: RMSE = {rmse:.5f}')
    results.append((ticker, rmse))

# put results in DataFrame
forecasted_results_df = pd.DataFrame(results, columns=['Ticker', 'RMSE'])
print(forecasted_results_df.head())

#save to csv 
forecasted_results_df.to_csv('forecasted_results.csv', index=False)