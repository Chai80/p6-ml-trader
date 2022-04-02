import numpy as np
import pandas as pd
from util import get_data
import matplotlib.pyplot as plt
import datetime as date
import warnings
warnings.filterwarnings("ignore")

'''
Choose 5 indicators to create
1. SMA
2. EMA
3. Volatility
4. Momentum
'''
def author():
    return 'wchai8'

def compute_sma(prices, movingWindow):
    # calculate simple moving average from prices
    SMA = prices.rolling(window=movingWindow,center=False).mean()
    # calculate the price to SMA ratio (PSR)
    PSR = prices / SMA - 1

    return SMA, PSR

def MACD(prices):
    ema_short = prices.ewm(span=12, adjust=False).mean()
    ema_long = prices.ewm(span=26, adjust=False).mean()
    MACD = ema_long - ema_short
    signal = MACD.ewm(span=9, adjust=False).mean()
    return MACD, signal

def getVolatility(prices, movingWindow):
    volatility = prices.rolling(window=movingWindow, center=False).std()
    return volatility



def get_BB(prices, moving_window):
    rolling_mean = prices.rolling(window=moving_window).mean()
    rolling_std = prices.rolling(window=moving_window).std()
    upperband = rolling_mean + (2 * rolling_std)
    lowerband = rolling_mean - (2 * rolling_std)
    BBP = (prices - lowerband) / (upperband - lowerband)
    return upperband, lowerband, BBP

def normalize(prices):
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    return prices / prices.ix[0, :]

def getMomentum(prices, moving_window):
    momentum = prices / prices.shift(moving_window) - 1
    return momentum

def getEMA(price, movingWindow):
    emaDF = pd.DataFrame.copy(price)
    emaDF.shift(movingWindow - 1)
    emaDF.iloc[movingWindow-1] = price.ix[0,0]
    k = 2/(movingWindow + 1)
    for i in range(emaDF.shape[0]):
        if i >= movingWindow:
            value = price.iloc[i] *k
            value2 = emaDF.iloc[i-1] * (1-k)
            emaDF.iloc[i] = value + value2
    emaDF[:movingWindow-1] = np.nan
    emaDF.fillna(method = "bfill", inplace = True)
    return emaDF

def compute_indicators():
    syms = ['JPM']
    startDate = date.datetime(2008, 1, 1)
    endDate = date.datetime(2009, 12, 31)
    dates = pd.date_range(startDate, endDate)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols




if __name__ == "__main__":
        compute_indicators()