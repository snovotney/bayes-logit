import numpy as np
import theano
import theano.tensor as T


def logistic(x):
    return 1.0 / (1.0 + T.exp(-x))


class LogisticRegression(object):
    def __init__(self, X, Y, alpha=0.1):

        n_in = X.shape[1]
        n_out = 1

        self.x = T.dmatrix("x")
        self.y = T.dmatrix("y")
        self.theta = theano.shared(value=np.zeros((n_in, n_out)), name='w')
        self.alpha = T.scalar("alpha")
        
        self.prob = logistic(T.dot(self.x,self.theta))
        self.pred = self.prob > 0.5

        self.lklhd = -self.y * T.log(self.prob) - (1 - self.y) * T.log(1 - self.prob)
        self.cost = self.lklhd.mean()

        self.gtheta = T.grad(self.cost, self.theta)
        
        self.update = theano.function(
            inputs=[self.x, self.y, self.alpha],
            outputs=[self.cost, self.theta],
            updates=[ [ self.theta, self.theta - self.alpha * self.gtheta]]
            )

        self.predict = theano.function(inputs=[self.x], outputs=self.pred)
        
class LogisticRegressionPrior(object):

    def  __init__(self,models):
        print( "loaded logit prior")
