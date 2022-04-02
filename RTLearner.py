import numpy as np


# RT learner takes 2 arguments: leaf_size and verbose
# leaf_size is a hyperparameter that defines the maximum number of samples to be aggregated at a leaf.
# when verbose is true, code generates output to screen for debugging purporses
# When the tree is constructed recursively, the data should be aggregated into a leaf
#Xtrain and Xtest should be ND arrays(Numpy objects)
#Each row represents a set of feature values
#Colums are the features and the rows are individual example isntances
#ypred and ytrain are single dimensions ND arrays
#ypred is the prediciton based on the given feature dataset
####We define “best feature to split on” as the feature (Xi) that has the highest absolute value correlation with Y.
#we are building a regression tree

#Add_evidence should only be called once with training data. Once trained, I suspect that if you want to call it again it would build a new tree using the provided training data.

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.tree = None
        self.verbose = verbose
        self.leaf_size = leaf_size
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        return 'wchai8'  # username


    def buildTree(self, xTraining, yTraining):

        if (xTraining.shape[0] <= self.leaf_size) or (np.all(yTraining == yTraining[0])):
            return np.array([[-1, np.mean(yTraining), -1, -1]])
        if np.std(yTraining) == 0.0:
            return np.array([[self.leaf, yTraining.mean(), self.NA, self.NA]])

        ###RT Learner - the choice of feature to split on must be random
        # We need to find the feature to split on through this function
        featureIndex = np.random.randint(0, xTraining.shape[1])

        random_split = (xTraining[np.random.randint(xTraining.shape[0]), featureIndex] + xTraining[np.random.randint(xTraining.shape[0]), featureIndex]) / 2.0

        if np.all(xTraining[:, featureIndex] <= random_split):
            return np.array([[-1, np.mean(yTraining), -1, -1]])

        checkLeft = xTraining[:, featureIndex] <= random_split
        checkRight = xTraining[:, featureIndex] > random_split

        leftBranch = self.buildTree(xTraining[checkLeft], yTraining[checkLeft])
        rightBranch = self.buildTree(xTraining[checkRight], yTraining[checkRight])
        root = np.array([featureIndex, random_split, 1, leftBranch.shape[0] +1])
        return np.vstack((root, leftBranch, rightBranch))

    def add_evidence(self, xTraining, yTraining):
        self.tree = self.buildTree(xTraining, yTraining)
        return self.tree

    def query(self, xTest):

        result = []
        # loop through all the queries
        for x in xTest:
            index = 0
            isLeaf = False
            while not isLeaf:
                check = int(self.tree[index][0])
                if check == -1:
                    result.append(self.tree[index][1])
                    isLeaf = True
                ##check factor left
                if x[check] <= self.tree[index][1]:
                    index += int(self.tree[index][2])
                else:
                    index += int(self.tree[index][3])

        return np.array(result)


if __name__=="__main__":
    print ("y")



