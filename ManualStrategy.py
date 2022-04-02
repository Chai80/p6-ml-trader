import pandas as pd
#pd.set_option("display.max_rows", None, "display.max_columns", None)
import numpy as np
import sys
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data
import csv

np.set_printoptions(threshold=506)

from indicators import compute_sma, get_BB, getMomentum, getVolatility

class ManualStrategy(object):

    def __init__(self):
        pass


    def benchmark(self, symbol, sd, ed):
        start_date = sd
        end_date = ed
        dates = pd.date_range(start_date, end_date)
        symbols = symbol
        prices = get_data([symbols], dates)
        del prices["SPY"]
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        prices = prices / prices.iloc[0]
        share = [1000, -1000]
        date = [prices.index[0], prices.index[len(prices.index) - 1]]
        df = pd.DataFrame(data=share, index=date, columns=['JPM'])
        return df

    def testPolicy(self, symbol , sd, ed, sv=100000):
        flag = 1
        moving_window = 21
        symbol = 'JPM'
        prices = get_data([symbol], pd.date_range(sd, ed))

        current = 0

        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        prices = prices / prices.iloc[0, :]

        trades = pd.DataFrame(columns=['Shares'], index=prices.index)
        sma, priceSMARatio = compute_sma(prices, moving_window)

        upperband, lowerband, BBP = get_BB(prices, moving_window)
        volatility = getVolatility(prices, moving_window)

        orders = prices.copy()
        orders[:] = 0

        for index in range(prices.shape[0]):
            i = prices.index[index]

            if flag == -1:

                if priceSMARatio.loc[i, symbol] < 0.5 or BBP.loc[i, symbol] < 0.2 or volatility.loc[i, symbol] < 0.025:
                    trades.loc[i] = 1000 - current
                    current = 1000
                    flag = 1

            elif flag == 1:
                if priceSMARatio.loc[i, symbol] > 1.4 or BBP.loc[i, symbol] > 0.8 or volatility.loc[i, symbol] > 0.075:
                    trades.loc[i] = -1000 - current
                    current = -1000
                    flag = -1

        trades.columns = [symbol]
        trades.fillna(0, inplace=True)
        return trades





def getPortfolioStats(port_val, rfr=0, sf=252):
    #Daily Returns
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]


    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(sf) * np.mean(adr - rfr) / sddr

    return cr, adr, sddr, sr


def author():
    print ("wchai8")




def Result() :
    ms = ManualStrategy()
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbol = 'JPM'

    # Manual Strategy
    tradesManual = ms.testPolicy(symbol = symbol, sd  = start_date, ed = end_date, sv = 100000)
    #print(tradesManual)

    portvalsManual = compute_portvals(tradesManual, 100000, 9.95, 0.005)
    portvalsManualNormalized = portvalsManual / portvalsManual.iloc[0]


    #cum_ret_man, avg_daily_ret_man, std_daily_ret_man, sharpe_ratio_man = getPortfolioStats(portvalsManualNormalized)


    # Benchmark Strategy
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    del prices["SPY"]

    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')
    pricesNormalized = prices/prices.iloc[0]

    benchmarkPrices = prices[symbol]
    tradesBenchmark = np.zeros(len(benchmarkPrices.index))
    tradesBenchmark[0] = 1000
    tradesBenchmark = pd.DataFrame(data=tradesBenchmark, index=benchmarkPrices.index, columns=[symbol])


    portvalsBenchmark = compute_portvals(tradesBenchmark, start_val=100000, commission=9.95, impact=0.005)

    # Normalizing PortFolio Values
    portvalsBenchmarkNormalized = portvalsBenchmark / portvalsBenchmark.iloc[0]

    #cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = getPortfolioStats(portvalsBenchmarkNormalized)


    plt.figure(figsize = (15,7.5))
    top = plt.subplot2grid((5, 1), (0, 0), rowspan=3, colspan=1)

    for index, marks in tradesManual.iterrows():

        if tradesManual.loc[index, symbol] > 0:
            plt.axvline(x=index, color='blue', linestyle='dashed', alpha = .9)
        elif tradesManual.loc[index, symbol] < 0:
            plt.axvline(x=index, color='black', linestyle='dashed', alpha = .9)

    volatility = getVolatility(pricesNormalized, 21)

    top.xaxis_date()
    top.plot(portvalsManualNormalized, lw = 2, color = 'red', label = 'Manual Strategy')
    top.plot(portvalsBenchmarkNormalized, lw=1.2, color='green', label='Benchmark')
    top.set_title("Manual Strategy vs Benchmark Strategy - In Sample")

    upperband, lowerband, BBP = get_BB(prices, 21)

    bottom = plt.subplot2grid((5, 1), (3, 0), rowspan=2, colspan=1, sharex=top)
    bottom.plot(BBP, color='olive', lw=1, label="Bollinger Band %")
    bottom.axhline(y=0.8, color='grey', linestyle='--', alpha=0.5)
    bottom.axhline(y= 0.2, color='grey', linestyle='--', alpha=0.5)
    bottom.set_ylabel('BBP %')
    bottom.legend()

    top.legend(loc = 'lower right')
    top.axes.get_xaxis().set_visible(False)
    top.set_ylabel("Normalized Portfolio Value")
    plt.xlim(start_date, end_date)
    plt.savefig('MS_InSample.png')
    plt.clf()


    figBBPIn = plt.figure(figsize=(12, 6.5))
    top2 = plt.subplot2grid((6, 1), (0, 0), rowspan=3, colspan=1)
    top2.plot(volatility, lw=2, color='blue', label=' Volatility')
    top2.xaxis_date()
    top2.legend()
    top2.set_title('Volatility Indicator')
    top2.axhline(y=0.025, color='grey', linestyle='--', alpha=0.5)
    top2.axhline(y=0.075, color='grey', linestyle='--', alpha=0.5)
    figBBPIn.savefig('Volatility In Sample')
    plt.clf()




    #Out of Sample Code
    ms = ManualStrategy()
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    symbol = "JPM"

    prices2 = get_data([symbol], pd.date_range(start_date, end_date))
    del prices2["SPY"]
    prices2 = prices2.fillna(method='ffill')
    prices2 = prices2.fillna(method='bfill')
    prices2Normalized = prices2/prices2.iloc[0]


    volatility2 = getVolatility(prices2Normalized, 21)
    # Manual Strategy
    tradesManual = ms.testPolicy(symbol, sd=start_date, ed=end_date, sv=100000)
    portvalsManual = compute_portvals(tradesManual, start_val=100000, commission=9.95, impact=0.005)

    # Normalizing PortFolio Values
    portvalsManualNormalized = portvalsManual / portvalsManual.iloc[0]

    #cum_ret_man2, avg_daily_ret_man2, std_daily_ret_man2, sharpe_ratio_man2 = getPortfolioStats(portvalsManualNormalized)


    # Benchmark Strategy
    benchmarkPrices = prices[symbol]
    tradesBenchmark = np.zeros(len(benchmarkPrices.index))
    tradesBenchmark[0] = 1000

    tradesBenchmark = ms.benchmark(symbol=symbol, sd=start_date, ed=end_date)
    portvalsBenchmark = compute_portvals(tradesBenchmark, start_val=100000, commission=9.95, impact=0.005)

    # Normalizing PortFolio Values
    portvalsBenchmarkNormalized = portvalsBenchmark / portvalsBenchmark.iloc[0]

    #cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = getPortfolioStats(portvalsBenchmarkNormalized)


    plt.figure(figsize=(18, 6.5))
    top = plt.subplot2grid((5, 1), (0, 0), rowspan=2, colspan=1)

    for index, marks in tradesManual.iterrows():

        if tradesManual.loc[index, symbol] > 0:
            plt.axvline(x=index, color='blue', linestyle='dashed', alpha=.9)
        elif tradesManual.loc[index, symbol] < 0:
            plt.axvline(x=index, color='black', linestyle='dashed', alpha=.9)

    top.xaxis_date()
    top.grid(True)
    top.plot(portvalsManualNormalized  , lw=2, color='red', label='Manual Strategy')
    top.plot(portvalsBenchmarkNormalized, lw=1.2, color='green', label='Benchmark')
    top.set_title("Manual Strategy vs Benchmark Strategy - Out Of Sample")

    upperband2, lowerband2, BBP2 = get_BB(prices2, 21)
    bottom = plt.subplot2grid((5, 1), (3, 0), rowspan=2, colspan=1, sharex=top)
    bottom.plot(BBP2, color='olive', lw=1, label="Bollinger Band %")
    bottom.axhline(y=0.8, color='grey', linestyle='--', alpha=0.5)
    bottom.axhline(y=0.2, color='grey', linestyle='--', alpha=0.5)
    bottom.set_ylabel('BBP  %')
    bottom.legend()
    top.legend(loc='lower right')
    top.axes.get_xaxis().set_visible(False)
    top.set_ylabel("Normalized Portfolio Value")
    plt.xlim(start_date, end_date)
    plt.savefig('MS_OutSample.png')
    plt.clf()


    figBBPOut = plt.figure(figsize = (12,6.5))
    top3 = plt.subplot2grid((6, 1), (0, 0), rowspan=3, colspan=1)
    top3.plot(volatility2, lw = 2, color = 'blue', label = 'volatility')
    top3.xaxis_date()
    top3.legend()
    top3.set_title('Volatility Indicator')
    top3.axhline(y = 0.025, color = 'grey', linestyle = '--', alpha = 0.5)
    top3.axhline(y = 0.075, color = 'grey', linestyle = '--', alpha = 0.5)
    figBBPOut.savefig('Volatility Out of Sample')
    plt.clf()




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


        trade.loc[date, 'Cash'] = trade.loc[date, 'Cash'] - commission - (stock_price * shares)

        holding = trade.cumsum()
        holding_value = holding * prices
        portvals = holding_value.sum(axis=1)

    return portvals


if __name__ == "__main__":
    Result()