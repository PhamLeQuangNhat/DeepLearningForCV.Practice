import numpy as np 

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize weights matrices, then store 
        # network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the
            # number of nodes in each respective layer together,
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
        
        # the last two layers are a special case where the input
        # connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string represents nn
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a
        # given input value
        return 1.0 / (1 + np.exp(-x))    

    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function,
        # x = sigmoid function
        return x * (1 - x)   

    def fit(self, X, y, epochs=1000, verbose=100):
        # insert a column of 1’s as the last entry 
        # in the feature matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over number of epochs
        for epoch in np.arange(0,epochs):
            # loop over each individual data point and train
            # our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see display a training update
            if epoch == 0 or (epoch + 1) % verbose == 0:
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                    epoch + 1, loss))

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over layers in the network
        for layer in np.arange(0, len(self.W)):
            # net input
            #print(A[layer].shape)
            #print(self.W[layer].shape)
            net  = A[layer].dot(self.W[layer])

            # net output
            out = self.sigmoid(net)

            A.append(out)
        
        # BACKPROPAGATION
        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        # loop over the layers
        for layer in np.arange(0,len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        # initialize the output prediction as the input features
        p = np.atleast_2d(X)

        # check to see if bias column should be added
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return predicted value
        return p
    
    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute
        # the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # return the loss
        return loss
