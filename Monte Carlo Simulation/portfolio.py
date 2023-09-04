"""
Implement Monte Carlo Method to simulate a portfolio 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# import data

def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stocks = ['TSLA','AAPL','MSFT','SOFI', 'CVNA']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo method
# number of simulations

mc_sims = 10
T = 50 #time frame in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)

print(meanM)
meanM = meanM.T
print(meanM)
portfolio_simulations = np.full(shape=(T, mc_sims), fill_value=0.0) #empty array

initial_mv = 100000

L = np.linalg.cholesky(covMatrix)
    
for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_simulations[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1) * initial_mv

plt.plot(portfolio_simulations)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulations of my Portfolio')

plt.show()