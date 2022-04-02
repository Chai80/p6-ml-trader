import StrategyLearner as sl
import ManualStrategy
from ManualStrategy import compute_portvals
import datetime as dt
import random
import numpy as np
random.seed(1481090000)
np.random.seed(1481090000)
import matplotlib.pyplot as plt
import pandas as pd

from util import get_data



def author():
    return 'wchai8'

def getPortfolioStats(port_val, rfr=0, sf=252):
    #Daily Returns
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]

    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()

    return cr, adr, sddr

def compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005):
    orders = df_trades
    orders.sort_index(inplace=True)
    start_date = orders.index.min()
    end_date = orders.index.max()

    stocks = list(df_trades.columns)

    prices = get_data(stocks, pd.date_range(start_date, end_date))
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)

    prices['Cash'] = 1

    trade = pd.DataFrame(np.zeros(prices.shape), columns=prices.columns, index=prices.index)
    trade.iloc[0, -1] = start_val

    for index, row in orders.iterrows():
        date = index
        stock = 'JPM'
        shares = row[stock]
        stock_price = prices.loc[index, stock]

        if shares > 0:
                trade.loc[date, stock] = trade.loc[date, stock] + shares
                stock_price = (1 + impact) * stock_price


        else:
                trade.loc[date, stock] = trade.loc[date, stock] + shares
                stock_price = (1 - impact) * stock_price

            # accounting market impact
        trade.loc[date, 'Cash'] = trade.loc[date, 'Cash'] - commission - (stock_price * shares)

        holding = trade.cumsum()
        holding_value = holding * prices
        portvals = holding_value.sum(axis=1)

        cr, addr, sddr = getPortfolioStats(portvals)

    return portvals, cr, addr, sddr

def test_code():
    ms = ManualStrategy.ManualStrategy()
    symbol = 'JPM'
    startDate = dt.datetime(2008, 1, 1)
    endDate = dt.datetime(2009, 12, 31)

    tradesManual = ms.testPolicy(symbol, startDate, endDate, 100000)

    manualPortvals, manCR, manMean, manSTD = compute_portvals(tradesManual, 100000, commission=0.0, impact=0.0)

    learner = sl.StrategyLearner(verbose=False, impact=0.0)
    learner.add_evidence(symbol=symbol, sd=startDate, ed=endDate, sv=100000)
    tradesLearner = learner.testPolicy(symbol, startDate, endDate, 100000)

    learnerPortvals, learnerCR, learnerMean, learnerSTD = compute_portvals(tradesLearner, 100000, commission=0.0, impact=0.0)

    normalizedManual = manualPortvals / manualPortvals.iloc[0]
    normalizedLearner = learnerPortvals / learnerPortvals.iloc[0]

    ##bench mark
    prices = get_data([symbol], pd.date_range(startDate, endDate))
    del prices["SPY"]

    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')

    benchmarkPrices = prices[symbol]
    tradesBenchmark = np.zeros(len(benchmarkPrices.index))
    tradesBenchmark[0] = 1000
    tradesBenchmark = pd.DataFrame(data=tradesBenchmark, index=benchmarkPrices.index, columns=[symbol])
    portvalsBenchmark, benchMarkCR, benchmarkMean, benchMarkSTD = compute_portvals(tradesBenchmark, start_val=100000, commission=9.95, impact=0.005)
    normBenchmark = portvalsBenchmark/portvalsBenchmark.iloc[0]


    plt.figure(10)
    plt.title("Experiment 1: Manual Strategy vs. Strategy")
    plt.ylabel("Normalized Portfolio Value")
    plt.xlabel("Dates")

    plt.plot(normalizedManual, 'r', label="Manual")
    plt.plot(normalizedLearner, 'b', label="Strategy Learner")
    plt.plot(normBenchmark, 'g', label = "BenchMark")
    plt.legend()
    plt.xticks(size = 8)
    plt.savefig("Experiment 1.png")
    plt.clf()

if __name__ == '__main__':
    test_code()
