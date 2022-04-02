import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import StrategyLearner as sl
from experiment1 import compute_portvals

def author(self):
    return 'wchai8'


def Portfolio_Statistics(portvals):
    daily_returns = portvals.copy()
    daily_returns[1:] = (portvals[1:] / portvals[:-1].values) - 1
    daily_returns = daily_returns[1:]

    cum_ret = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(252.0) * (avg_daily_ret / std_daily_ret)

    return cum_ret,avg_daily_ret,std_daily_ret,sharpe_ratio
def test_code():
    startDate = dt.datetime(2008, 1, 1)
    endDate = dt.datetime(2009, 12, 31)
    symbol = "JPM"


    # Impact 0.0
    learner = sl.StrategyLearner(verbose=False, impact=0.0)
    learner.add_evidence(symbol=symbol, sd=startDate, ed=endDate, sv=100000)
    trades = learner.testPolicy(symbol=symbol, sd=startDate, ed=endDate, sv=100000)
    portVals, cr, addr, sddr = compute_portvals(trades, start_val=100000, commission=0.0, impact=0.0)
    portValsNormalized = portVals / portVals.iloc[0]

    # Impact 0.001
    learner2 = sl.StrategyLearner(verbose=False, impact=0.001)
    learner2.add_evidence(symbol=symbol, sd=startDate, ed=endDate, sv=100000)
    trades2 = learner2.testPolicy(symbol=symbol, sd=startDate, ed=endDate, sv=100000)
    portVals2, cr2, addr2, sddr2 = compute_portvals(trades2, start_val=100000, commission=0.0, impact=0.002)
    portVals2Normalized = portVals2 / portVals2.iloc[0]

    # Impact 0.01
    learner3 = sl.StrategyLearner(verbose=False, impact=0.01)
    learner3.add_evidence(symbol=symbol, sd=startDate, ed=endDate, sv=100000)
    trades3 = learner3.testPolicy(symbol=symbol, sd=startDate, ed=endDate, sv=100000)
    portVals3, cr3, addr3, sddr3 = compute_portvals(trades3, start_val=100000, commission=0.0, impact=0.05)
    portVals3Normalized = portVals3 / portVals3.iloc[0]

    # Impact 0.1
    learner4 = sl.StrategyLearner(verbose=False, impact=0.1)
    learner4.add_evidence(symbol=symbol, sd=startDate, ed=endDate, sv=100000)
    trades4 = learner4.testPolicy(symbol=symbol, sd=startDate, ed=endDate, sv=100000)
    portVals4, cr4, addr4, sddr4 = compute_portvals(trades4, start_val=100000, commission=0.0, impact=0.1)
    portVals4Normalized = portVals4 / portVals4.iloc[0]

    plt.figure(9)
    plt.title("Strategy Learner with different impact values")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.plot(portValsNormalized, label="Impact 0.0")
    plt.plot(portVals2Normalized, label="Impact 0.001")
    plt.plot(portVals3Normalized, label="Impact 0.01")
    plt.plot(portVals4Normalized, label="Impact 0.1")
    plt.legend()
    plt.xticks(size =8)
    plt.savefig("Experiment2.png")
    plt.clf()



if __name__ == "__main__":

    np.random.seed(333)
    test_code()