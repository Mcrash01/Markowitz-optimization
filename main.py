# require "pip install yfinance"

import yfinance as yf
import pandas as pd
from datetime import datetime
stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

start_date = '2022-01-01'

end_date = datetime.now().strftime('%Y-%m-%d')

stocks_data = yf.download(stock_symbols, start=start_date, end=end_date)['Adj Close']

returns = stocks_data.pct_change()
returns = returns.iloc[1:]
print("Stock Returns:")
print(returns)
covariance_matrix = returns.cov()
print("Covariance Matrix:")
print(covariance_matrix)
mean_returns = returns.mean()

variance_returns = returns.var()

print("Mean Returns:")
print(mean_returns)

print("\nVariance of Returns:")
print(variance_returns)