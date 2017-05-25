import sys
import numpy as np
import timeit

sys.path.append('lib')

from bayesme2 import *
from dataset import *
from sklearn.linear_model import LogisticRegression as LogIt

num_sets = 3
batch_size = 50 #TODO: variable batch size depending on amoutn of data?
num_epochs = 10
learning_rate = 0.1
# read in the N different training sets
# X is of dim (K x M x Nk)
#    K = # of unique domains
#    M = # of feature dimensions
#    Nk = # of training examples for dataset k
#
# Y is of dim (K x 1 x Nk)
#    Only difference is that the feature space is a binary variable
#
data=ToyDataSet()
trainX, trainY, testX, testY = data.load(num_samples=[1000] * num_sets, num_sets=num_sets, means=[-1,0,1])

models = []
for i in range(num_sets):
    models.append(LogisticRegression(trainX[i],trainY[i]))


#run one epoch of training for a specific model
def sgd_one_epoch(model, X, Y, Xval, Yval, learning_rate):
    num_train_batches = X.shape[0] // batch_size
    num_val_batches = Xval.shape[0] // batch_size

    avg_cost = 0.
    val_acc = 0.
    
    theta = None
    for mb_index in range(num_train_batches):
        cost, thetaBest =  model.update(X[mb_index * batch_size : (mb_index + 1) * batch_size ],
                                   Y[mb_index * batch_size : (mb_index + 1) * batch_size ],
                                   learning_rate)

        theta = thetaBest
        avg_cost += cost

    for mb_index in range(num_val_batches):
        pred = model.predict(Xval[mb_index * batch_size : (mb_index + 1) * batch_size])
        acc = np.mean( pred == Yval[mb_index * batch_size : (mb_index + 1) * batch_size])
        val_acc += acc

    return(avg_cost / num_train_batches, 100 * val_acc / num_val_batches, theta.flatten())

    
    

    
#
for epoch in range(num_epochs):
    for i in range(num_sets):
        train_cost, test_acc, theta = sgd_one_epoch(models[i],trainX[i],trainY[i],testX[i], testY[i], learning_rate)
        print("{:2d} model {:2d}: cost={:5.2f} test-acc: {:5.2f} ".format(epoch+1,i, train_cost, test_acc), theta)
        



