import numpy as np
import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, X, Y,batch_size=5, loss_function='lklhd'):

        print("creating model")

        self.borrow=False                
        self.batch_size=batch_size
        self.learning_rate=0.13

        # number of features
        self.n_in = X.shape[1]
        self.n_out = 2 #always binary
        
        self.index = T.lscalar()

        # X is features and Y is labels
        self.X = theano.shared(np.asarray(X, dtype=theano.config.floatX),borrow=self.borrow)
        self.Y = theano.shared(np.asarray(Y, dtype=theano.config.floatX),borrow=self.borrow)

        # fix Y so that it can be passed to functions
        self.Y = self.Y.flatten()
        self.Y = T.cast(self.Y, 'int32')

        # params
        self.W = theano.shared(value = np.zeros((self.n_in, self.n_out), dtype=theano.config.floatX),
                               name='W',borrow=self.borrow)
        self.b = theano.shared(value = np.zeros((self.n_out,), dtype=theano.config.floatX),
                               name='b',borrow=self.borrow)

        self.params = [self.W, self.b]        

        # these are symbolic variables
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        self.p_y_given_x = T.nnet.softmax(T.dot(self.x, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Validation graph
        self.validate =  theano.function(
            inputs=[self.index],
            outputs=self.errors(self.y),
            givens={
                self.x: self.X[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.Y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        # setup the training graph

        if loss_function == 'lklhd':
            self.cost = self.lklhd(self.y)
        elif loss_function == 'L2':
            self.cost = self.lklhd_L2(self.y)
        elif loss_function == 'gaussian':
            self.cost = self.lklhd_gaussian_prior(self.y)
        else:
            raise Exception("Unknown loss function", loss_function)

        self.g_W = T.grad(cost=self.cost, wrt=self.W)
        self.g_b = T.grad(cost=self.cost, wrt=self.b)

        self.updates = [(self.W, self.W - self.learning_rate * self.g_W),
                        (self.b, self.b - self.learning_rate * self.g_b)]

        self.train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x: self.X[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.Y[self.index * self.batch_size: (self.index + 1) * self.batch_size],
            }
        )        

    
    # negative log likelihood
    def lklhd(self,Y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(Y.shape[0]), Y])

    def lklhd_L2(self):
        pass

    def lklhd_gaussian_prior(self):
        pass
            
        

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
    def train(self, index, prior):
        self.train_model(index)


    def validate(self, index):
        return self.validate(index)


    
        
class LogisticRegressionPrior(object):

    def  __init__(self,models):
        print( "loaded logit prior")
