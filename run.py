import sys
import numpy as np
import timeit

sys.path.append('lib')

from scipy.optimize import minimize
from bayesme import *
from dataset import *
from sklearn.linear_model import LogisticRegression as LogIt


num_sets = 5
batch_size = 128  #TODO: variable batch size depending on amount of data?
num_epochs = 50
learning_rate = 0.01

learning_rate_decay = 0.95
tol = 0.98 # new accuracy must be less than old acc * tolerance to keep training
sigma2 = [ 0.001 * 2**x for x in range(num_sets) ]
print(sigma2)

psigma = 0.01 / num_sets


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
trainX, trainY, testX, testY = data.load(num_samples=[10000] * num_sets, num_sets=num_sets, means=[-1, -0.5, 0, 0.5, 1])


models = []
for i in range(num_sets):
    models.append(LogisticRegression(trainX[i],trainY[i], loss='gaussian'))

pi = LogisticRegressionPrior(trainX[0].shape[1],sigma2=psigma)

#run one epoch of training for a specific model
def sgd_one_epoch(model, X, Y, Xval, Yval, learning_rate, theta0, sigma):

    
    num_train_batches = X.shape[0] // batch_size
    num_val_batches = Xval.shape[0] // batch_size

    avg_cost = 0.
    avg_lklhd = 0.
    avg_prior = 0.
    val_acc = 0.
    theta = None

    #iterate over all the training data doing minibatch updates
    for mb_index in range(num_train_batches):
        cost, lklhd, prior, thetaBest =  model.update(X[mb_index * batch_size : (mb_index + 1) * batch_size ],
                                                      Y[mb_index * batch_size : (mb_index + 1) * batch_size ],
                                                      learning_rate, theta0, sigma)
        theta = thetaBest
        avg_prior += prior
        avg_lklhd += lklhd
        avg_cost += cost

    #iterate over validation data to compute heldout accuracy
    for mb_index in range(num_val_batches):
        pred = model.predict(Xval[mb_index * batch_size : (mb_index + 1) * batch_size])
        acc = np.mean( pred == Yval[mb_index * batch_size : (mb_index + 1) * batch_size])
        val_acc += acc

    return(avg_cost / num_train_batches,
           avg_lklhd / num_train_batches,
           avg_prior / num_train_batches,
           100 * val_acc / num_val_batches,
           theta.flatten())

curr_theta = [0] * num_sets
best_theta = [0] * num_sets    
best_acc   = [0] * num_sets    # per-model best validation accuracy
best_epoch = [0] * num_sets  # per-model best  epoch id
update     = [True] * num_sets   # per-model boolean whether to keep training

# train models up to #of epochs w/validation-driven stopping criterion
for epoch in range(num_epochs):

    learning_rate *= learning_rate_decay

    if not any(update):
        break
    for i in range(num_sets):
        if update[i]:
            cost, lklhd, prior, test_acc, theta = sgd_one_epoch(models[i],
                                                                trainX[i],trainY[i],
                                                                testX[i], testY[i],
                                                                learning_rate,
                                                                pi.theta,
                                                                sigma2[i]
                                                        )
            curr_theta[i] = theta

#            print("{:2d} model {:2d}: cost={:5.5f} lklhd={:5.5f} prior={:5.5f} test-acc: {:5.5f} ".format(epoch+1,i, cost, lklhd, prior,test_acc), theta)
            print("{:2d} model {:2d}: cost={:5.5f} lklhd={:5.5f} prior={:5.5f} test-acc: {:5.5f} ".format(epoch+1,i, cost, lklhd, prior,test_acc))            

            if test_acc > best_acc[i]:
                best_acc[i] = test_acc
                best_epoch[i] = epoch
                best_theta[i] = theta
                
            # if accuracy went down and the best accuracy was seen more than two epochs ago
            if test_acc <= best_acc[i] and (epoch - best_epoch[i]) > 1:
                update[i] = False
                print("no longer updating model", i)

    # update priors
    pi.update(best_theta, sigma2)
#    print("prior", pi.theta.flatten())

        

print()

print("logMeanWeight of prior={:5.2f}".format( np.log(np.abs(np.mean(pi.theta.flatten())))))
for i in range(num_sets):
    print("acc:{:5.2f} best epoch:{:3d} logPenalty:{:5.2f} logMeanWeightDiff:{:5.2f} sigma:{:.5f}" .format(
        best_acc[i], 1+best_epoch[i],
        np.log(np.mean(pi.theta.flatten() - best_theta[i])**2),
        np.log(np.abs(np.mean(best_theta[i]))),
        sigma2[i]
    ))




