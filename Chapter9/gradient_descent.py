from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt 
import numpy as np 
import argparse

# construct the argument parse and parse the arguments
def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=float, default=100,
                    help="Number of epochs")
    ap.add_argument("-a", "--alpha",type=float, default=0.01,
                    help="learning rate")
    args = vars(ap.parse_args())
    return args

def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    # take the dot product between features and weight matrix
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds

def main():

    args = option()

    # generate a 2-class classification problem with 1,000 data points,
    # where each data point is a 2D feature vector
    (X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                        cluster_std=1.5, random_state=1)
    print(X.shape)                  
    y  =  y.reshape((y.shape[0], 1))

    # insert a column of 1’s as the last entry in the feature matrix
    X = np.c_[X,np.ones((X.shape[0]))]
    print(X.shape)

    # split training:75%, testing:25%
    (trainX, testX, trainY, testY) = train_test_split(X, y,
                                     test_size=0.5, random_state=42)
    
    # initialize weight matrix and list of losses
    print("[INFO] training ...")
    W = np.random.rand(X.shape[1],1)
    losses = []

    # loop over number of epochs
    for epoch in range(0, args["epochs"]):
        preds = sigmoid_activation(trainX.dot(W))

        error = preds - trainY
        loss = np.sum(error**2)
        losses.append(loss)
        gradient = trainX.T.dot(error)
        W += -args["alpha"] * gradient

        # check to see if an update should be displayed
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),
                  loss))

    # evaluate model
    print("[INFO] evaluating ...")
    preds = predict(testY, W)
    print(classification_report(testY,preds))

    # plot testing classification data
    plt.style.use("ggplot")
    plt.figure()
    plt.title("Data")
    plt.scatter(testX[:,0], testX[:,1], maker="o", c=testY, s=30)

    # construct a figure that plots the loss over time
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, args["epochs"]), losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()

if __name__ == '__main__':
    main()