import numpy as np
import theano
import theano.tensor as T

import scipy.optimize as so

def logistic(x):
    return 1.0 / (1.0 + T.exp(-x))


class LogisticRegression(object):
    def __init__(self, X, Y, alpha=0.1,loss='lklhd'):

        n_in = X.shape[1]
        n_out = 1

        self.x = T.dmatrix("x")
        self.y = T.dmatrix("y")
        self.theta = theano.shared(value=np.zeros((n_in, n_out)), name='w')
        self.theta0 = T.dmatrix("theta0")
        self.sigma2 = T.scalar("sigma2")
        self.alpha = T.scalar("alpha") #loss function

        
        self.prob = logistic(T.dot(self.x,self.theta))
        self.pred = self.prob > 0.5

        self.lklhd = -self.y * T.log(self.prob) - (1 - self.y) * T.log(1 - self.prob)
        self.prior = ((self.theta - self.theta0) ** 2) *  self.sigma2

        #MLE cost = sum p(yi|xi)
        if loss == 'lklhd':
            self.cost = self.lklhd

            self.gtheta = T.grad(self.cost, self.theta)

            self.update = theano.function(
                inputs=[self.x, self.y, self.alpha],
                outputs=[self.lklhd, self.theta],
                updates=[ [ self.theta, self.theta - self.alpha * self.gtheta]]
            )


        # MAP: cost = sum p(yi|xi) + p(theta|theta0)
        # 
        elif loss == 'gaussian':
            self.cost =  self.lklhd.mean() + self.prior.mean()
            self.gtheta = T.grad(self.cost, self.theta)            

            self.update = theano.function(
                inputs=[self.x, self.y, self.alpha, self.theta0, self.sigma2],
                outputs=[self.cost, self.lklhd.mean(), self.prior.mean(), self.theta],
                updates=[ [ self.theta, self.theta - self.alpha * self.gtheta]]
            )
        else:
            raise Exception("Unknown loss ", loss)
            

        self.predict = theano.function(inputs=[self.x], outputs=self.pred)
        
class LogisticRegressionPrior(object):

    def  __init__(self,ndim,mean=0,sigma2=1):
        self.theta = np.zeros((ndim,1))
        self.sigma2 = sigma2
        self.theta.fill(mean)
        
    def objective(self,x):
        total = 0
        for i in range(len(self.thetas)):
            total += (x - self.thetas[i]) * self.sigmas[i] 
        return np.array(total - x * self.sigma2)
        
    def update(self, thetas, sigmas):
        self.thetas = thetas
        self.sigmas = sigmas
        root = so.fsolve(self.objective,x0 = [0] * len(thetas[0]))
        self.theta = np.ndarray(shape=self.theta.shape,buffer=root)


        
