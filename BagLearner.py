import numpy as np
import RTLearner as rt
import random
from scipy import stats

class BagLearner(object):

    def __init__(self, learner= rt.RTLearner, kwargs={}, bags=20, boost=False, verbose=False):
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return 'wchai8'

    def add_evidence(self, dataX, dataY):

        #create new matrix
        newdataX = np.zeros([dataX.shape[0], dataX.shape[1] + 1])
        newdataX[:, 0:dataX.shape[1]] = dataX
        newdataX[:, -1] = dataY

        for x in range(self.bags):
            index = np.random.choice(dataX.shape[0], dataX.shape[0], replace=True)
            tempDataX = newdataX[index, :-1]
            tempDataY = newdataX[index, -1]
            self.learners[x].add_evidence(tempDataX, tempDataY)

    def query(self, xTest):
        # x[0].shape gives length of 1st row of an array
        # x.shape[0] gives the number of rows of array x
        if not self.learners:
            ##makes sure bag learner is trained before being queried
            return np.nan

        predictions = np.zeros([self.bags, xTest.shape[0]])
        for i in range(self.bags):
            predictions[i] = self.learners[i].query(xTest)
        ypredMode = stats.mode(predictions,axis = 0)[0]
        return ypredMode
if __name__ == "__main__":
    print ("success")