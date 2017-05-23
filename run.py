import sys
import numpy as np
import timeit

sys.path.append('lib')

from bayesme import *
from dataset import *
from sklearn.linear_model import LogisticRegression as LogIt

num_sets = 3

# train on the K different subsets
batch_size = 50
num_epochs = 200
patience = 5 # run this many epochs
patience_increase = 2 #wait this much longer before new best is found
improvement_threshold = 0.998 # relative improvement required

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
trainX, trainY, testX, testY = data.load(num_samples=[5000] * num_sets, num_sets=num_sets, means=[-1,0,1])

models = []

for i in range(num_sets):
    models.append(LogisticRegression(trainX[i],trainY[i]))

# priors should have the same dimensionality as the union of all the models
#    theta = 
#    K sigmas for each of the component models
prior = LogisticRegressionPrior(models)


# train sklearn logit to compare
#logit = LogIt()
#logit.fit(trainX[0], trainY[0])
#print(1-logit.score(testX[0],testY[0]))

best_validation_loss = np.inf
done_looping = False

start_time = timeit.default_timer()

#load the #of batches for each dataset
num_train_batches = [0] * num_sets
num_valid_batches = [0] * num_sets

for i in range(num_sets):
    num_train_batches[i] = trainX[i].shape[0] // batch_size
    num_valid_batches[i] = testX[i].shape[0] // batch_size

#TODO: add patience update
epoch = 0
weights = [ [] for i in range(num_sets)]
best_validation_loss = [ np.inf ] * num_sets
update_model = [True ] * num_sets

while (epoch < num_epochs and not done_looping):
    epoch = epoch + 1

    #for each model
    for i in range(num_sets):

        model = models[i]

        if update_model[i]:
            #do mini-batch updates
            for minibatch_index in range(num_train_batches[i]):
                minibatch_avg_cost = model.train(minibatch_index, prior)

            # compute zero-one loss on validation set
            this_validation_loss = model.validate(testX[i], testY[i], num_valid_batches[i])

            if this_validation_loss < best_validation_loss[i]:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss[i] *  \
                   improvement_threshold:
                    patience = max(patience, epoch + patience_increase)

                    best_validation_loss[i] = this_validation_loss
                    print('model %i, epoch %i, minibatch %i/%i, validation error %f %%' % (
                        i,
                        epoch,
                        minibatch_index + 1,
                        num_train_batches[i],
                        this_validation_loss * 100.
                    )
                    )

                    #store off best weights for inspection later
                    W = model.params[0].get_value()
                    b = model.params[1].get_value()
                    weights[i].append([W,b])                    
                else: #if we saw a gain less than the desired threshold, stop updating
                    update_model[i] = False
        

        #TODO: add prior update
        
        if patience <= epoch:
            done_looping = True
            break
        
end_time = timeit.default_timer()
print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))
# visualize
# save out the K models

    
