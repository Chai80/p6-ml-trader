import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import util as ut
import ManualStrategy as ms
import experiment1 as experiment1
import experiment2 as experiment2

def author():
    print ("wchai8")

if __name__ == "__main__":

    #In sample and Out Sample Manual Strategy
    ms.Result()

    #Calling experiment1 script

    np.random.seed(333)
    experiment1.test_code()

    # Calling experiment2 script

    np.random.seed(333)
    experiment2.test_code()