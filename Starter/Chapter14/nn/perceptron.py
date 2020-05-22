import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        # initialize weight matrix and store lr
        self.W = np.random.randn(N+1) / np.sqrt(N)
        self.alpha = alpha
    
    def step(self,x):
        # apply step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over number of epochs
        for epoch in np.arange(0, epochs):
            # loop over individual data point
            for (x, target) in zip(X,y):
                pred = self.step(np.dot(x,self.W))

                # only perform a weight update if 
                # prediction != target
                if pred != target:
                    error = pred - target

                    # update weight matrix
                    self.W += -self.alpha * error * x
    
    def predict(self, X, addBias=True):
        # ensure our input is a matrix
        X = np.atleast_2d(X)
        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1’s as the last entry in the feature
            # matrix (bias)
            X = np.c_[X, np.ones((X.shape[0]))]
        
        return self.step(np.dot(X,self.W))

    

      
