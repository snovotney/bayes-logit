class LogisticRegression(object):
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
        self.theta = [0]
        self.sigma = [1]
        
        print( "created logit")

class LogisticRegressionPrior(object):

    def  __init__(self,models):
        print( "loaded logit prior")
