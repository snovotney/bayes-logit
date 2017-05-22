import numpy as np
from sklearn.datasets import make_classification

class DataSet:

    numDomains = 0
    
    def __init__(self):
        pass


    def load(self):
        print( "loaded data")
        pass


# create 2-dim training data for logit
#
# sample from 2-dim gaussians
#
class ToyDataSet(DataSet):
    def load(self,num_samples=None,num_sets=2,means=None):
        X = []
        Y = []

        if num_samples == None:
            num_samples = [ 100 for x in range(num_sets)]
        if means == None:
            means = [ 0 for x in range(num_sets)]

        for i in range(num_sets):
            x,y = make_classification(n_samples=num_samples[i], n_features=2,
                                      n_informative=2,
                                      n_redundant=0,
                                      n_classes=2,
                                      shift=means[i],
            )
            X.append(x)
            Y.append(y)

            print("Set {}: mean:{:2f} var{:2f}".format(i+1, np.mean(x), np.var(x)))
        return X, Y
