import sys
import numpy as np
sys.path.append('lib')

from bayesme import *
from dataset import *
# read in the N different training sets
# X is of dim (K x M x Nk)
#    K = # of unique domains
#    M = # of feature dimensions
#    Nk = # of training examples for dataset k
#
# Y is of dim (K x 1 x Nk)
#    Only difference is that the feature space is a binary variable
#
data = ToyDataSet()
X, Y = data.load()

# read in hyper-params

# create K different models
models = ()

for i in range(data.numDomains):
    models.append(LogisticRegression(X[i],Y[i]))

# priors should have the same dimensionality as the union of all the models
#    theta = 
#    K sigmas for each of the component models
prior = LogisticRegressionPrior(models)

# train on the K different subsets
train(models, prior)

# visualize

# save out the K models


def train(models, prior):
    print( "Training")
