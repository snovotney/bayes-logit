import sys
import numpy as np
import timeit

sys.path.append('lib')

from bayesme import *
from dataset import *

num_sets = 3


def train(models, prior):
    print( "Training")


    
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
trainX, trainY, testX, testY = data.load(num_samples=[5000] * num_sets, num_sets=num_sets, means=(-1,0,1))

models = []

for i in range(num_sets):
    models.append(LogisticRegression(trainX[i],trainY[i]))

# priors should have the same dimensionality as the union of all the models
#    theta = 
#    K sigmas for each of the component models
prior = LogisticRegressionPrior(models)

# train on the K different subsets
batch_size = 50
num_epochs = 100
start_time = timeit.default_timer()

#load the #of batches for each dataset
num_train_batches = [0] * num_sets
num_valid_batches = [0] * num_sets

for i in range(num_sets):
    num_train_batches[i] = trainX[i].shape[0] // batch_size
    num_valid_batches[i] = testX[i].shape[0] // batch_size


validation_frequency = min(num_train_batches * 2) #TODO

#TODO: add patience update
epoch = 0

while (epoch < num_epochs):
    epoch = epoch + 1

    #for each model
    for i in range(num_sets):

        model = models[i]

        #do mini-batch updates
        for minibatch_index in range(num_train_batches[i]):
            minibatch_avg_cost = model.train(minibatch_index, prior)

            # compute iteration number
            iter = (epoch - 1) * num_train_batches[i] + minibatch_index

            #compute validation loss and print
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [model.validate(i) for i in range(num_valid_batches[i])]
                this_validation_loss = np.mean(validation_losses)

                print('model %i, epoch %i, minibatch %i/%i, validation error %f %%' % (
                    i,
                    epoch,
                    minibatch_index + 1,
                    num_train_batches[i],
                    this_validation_loss * 100.
                    )
                )
                

        # TODO: compute stopping function
        
        # update thetas
        #TODO: add prior update
        
end_time = timeit.default_timer()
print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))
# visualize

# save out the K models

    
