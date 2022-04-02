import datetime as dt
import pandas as pd
import numpy as np
from util import get_data
import BagLearner as bl
import RTLearner as rt
from indicators import compute_sma, get_BB, getVolatility

class StrategyLearner(object):

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Constructor method  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False,
                                     verbose=False)
        self.N = 5

    def author(self):
        print("wchai8")


    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000):


        prices = get_data([symbol], pd.date_range(sd, ed))

        del prices["SPY"]
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        prices = prices / prices.iloc[0, :]

        # Getting Indicators values
        sma, price_sma = compute_sma(prices, 21)
        upperband, lowerband, BBP = get_BB(prices, 21)
        volatility = getVolatility(prices, 21)

        # Put all indicators into Data Frame
        df1 = sma.rename(columns={symbol: 'SMA'})
        df2 = BBP.rename(columns={symbol: 'BBP'})
        df3 = volatility.rename(columns={symbol: 'Volatility'})

        indicators_df = pd.concat((df1, df2, df3), axis = 1)
        indicators_df.columns = ['SMA', 'BBP', 'Volatility']



        Xtrain = indicators_df[:-self.N].values

        yTrain = []
        for i in range(prices.shape[0] - self.N):

            ret = (prices.ix[i + self.N, symbol] - prices.ix[i, symbol]) / prices.ix[i, symbol]

            if ret < (-0.015 - self.impact):
                yTrain.append(-1)

            elif ret > (0.015 + self.impact):
                yTrain.append(1)

            else:
                yTrain.append(0)
        yTrain = np.array(yTrain)


        self.learner.add_evidence(Xtrain, yTrain)

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000):
        prices = get_data([symbol], pd.date_range(sd, ed))
        del prices["SPY"]

        prices= prices.fillna(method ='ffill')
        prices = prices.fillna(method = 'bfill')
        prices = prices/prices.iloc[0,:]

        sma, price_sma = compute_sma(prices, 21)
        upperband, lowerband, BBP = get_BB(prices, 21)
        volatility = getVolatility(prices, 21)

        indicators = pd.concat((sma,BBP,volatility),axis=1)
        indicators.columns = ['SMA','BBP', 'Volatility']
        indicators.fillna(0, inplace = True)

        Xtest = indicators.values

        # Querying the learner
        Ytest = self.learner.query(Xtest)

        # Creating the trades DataFrame
        trades = prices.copy()
        trades.loc[:] = 0

        holdings = 0
        for i in range(prices.shape[0]):
            index = prices.index[i]
            if Ytest[0][i] >= 0.6:
                trades.loc[index, :] = 1000 - holdings
            elif Ytest[0][i] <= -0.6:
                trades.loc[index, :] = -1000 - holdings
            else:
                trades.loc[index, :] = 0.0
            holdings += trades.loc[index, :]

        return trades






if __name__ == "__main__":
    print("One dok")