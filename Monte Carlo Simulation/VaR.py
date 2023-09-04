"""
Implement VaR using Monte Carlo Simulation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights)

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def random_z_score():
    return np.random.normal(0,1)

days = 20

def scenario_gain_loss(portfolio_value, porfolio_std_dev, z_score, days):
    return portfolio_value * portfolio_expected_return * days + portfolio_value * porfolio_std_dev * z_score * np.sqrt(days)

# tickers = ['SPY', 'XLK', 'QQQ', 'VTI', 'AAPL']
tickers = ['TSLA','AAPL','MSFT','SOFI', 'CVNA']

years = 15 #Get 10 years of data

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365 * years)

daily_adjusted_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, startDate, endDate)
    daily_adjusted_close_df[ticker] = data['Adj Close']

log_returns = np.log(daily_adjusted_close_df/daily_adjusted_close_df.shift(1))
log_returns = log_returns.dropna()

cov_matrix = log_returns.cov()
print(cov_matrix)

portfolio_value = 1000000
weights = np.array([1/len(tickers)] * len(tickers))

portfolio_expected_return = expected_return(weights, log_returns)
portfolio_Std_devitation = standard_deviation(weights, cov_matrix)

simulations = 10000
scenarioReturn = []

for i in range(simulations):
    z_score = random_z_score()
    scenarioReturn.append(scenario_gain_loss(portfolio_value, portfolio_Std_devitation, z_score, days))

# print(scenarioReturn)
    
confidence_interval = 0.95
VaR = -np.percentile(scenarioReturn, 100* (1 - confidence_interval))
print(VaR)

plt.hist(scenarioReturn, bins=50, density=True)
plt.xlabel("Scenario Gain / Loss($)")
plt.ylabel("Frequency")
plt.title(f"Distribution of Portfolio Gain / Loss Over {days} days")
plt.axvline(-VaR, color='r', linestyle='dashed',linewidth=2, label=f'VaR at {confidence_interval: .0%} confidence level')
plt.legend()
plt.show()