import numpy as np 
from nn.perceptron import Perceptron

# construct OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# define perceptron and train it 
print("[INFO] training perceptron ...")
model = Perceptron(X.shape[1], alpha=0.1)
model.fit(X, y, epochs = 20)

print("[INFO] testing perceptron...")

# loop over the data points
for (x, target) in zip(X, y):
    # make a prediction on the data point and display the result
    # to our console
    pred = model.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(
    x, target[0], pred))
