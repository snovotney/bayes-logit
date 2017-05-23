import numpy as np
from sklearn.datasets import make_classification
import theano

borrow = True
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
        Xtrain = []
        Ytrain = []

        Xtest = []
        Ytest = []

        if num_samples == None:
            num_samples = [ 100 for x in range(num_sets)]
        if means == None:
            means = [ 0 for x in range(num_sets)]

            
        for i in range(num_sets):

            cutoff = int(0.67 * num_samples[i])
            
            x,y = make_classification(n_samples=num_samples[i], n_features=2,
                                      n_informative=2,
                                      n_redundant=0,
                                      n_classes=2,
                                      shift=means[i],
            )


            Xtrain.append(x[0:cutoff])
            Ytrain.append(y[0:cutoff])
            Xtest.append(x[cutoff:])
            Ytest.append(y[cutoff:])
                        
            print("Set {}: #samples:{} mean:{:2f} var{:2f}".format(i+1, num_samples[i], np.mean(x), np.var(x)))
            
        return Xtrain, Ytrain, Xtest, Ytest
