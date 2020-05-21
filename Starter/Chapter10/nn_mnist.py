from nn.neuralnetwork import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn import datasets

# load the MNIST dataset
dataset = datasets.load_digits()

# apply min/max scaling to scale the
# pixel intensity values to the range [0, 1]
data = dataset.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],data.shape[1]))
labels = dataset.target

# split training: 75%, testing: 25%
(trainX, testX, trainY, testY) = train_test_split(data,
                                labels, test_size=0.25, random_state=42)

# convert labels as vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY  = lb.fit_transform(testY)

# train the network
print("[INFO] training network ...")
model = NeuralNetwork([trainX.shape[1], 32, 16, 10],alpha=0.5)
print("[INFO] {}".format(model))
model.fit(trainX, trainY, epochs=1000)

# evaluate network
print("[INFO] evaluating network...")
preds = model.predict(testX)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1)))
