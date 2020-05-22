"""use python Chapter10/keras_mnist.py --dataset datasets/handwritten/ --output Chapter10/image_mnist.png"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

import matplotlib.pyplot as plt 
import numpy as np 
import argparse

import os
import gzip

# construct the argument parse and parse the arguments
def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=True,
                help="path to input data")
    ap.add_argument("-o", "--output", required=True,
                help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())
    return args

def load_data(path, kind='train'):    
    """Load handwritten_MNIST data from path"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

def main():
    args = option()

    # load the MNIST dataset 
    trainX, trainY = load_data(args["dataset"])
    testX, testY = load_data(args["dataset"], kind='t10k')
    
    # scale the raw pixel intensities to the range [0, 1]
    trainX = trainX.astype("float") / 255.0
    testX  = testX.astype("float") / 255.0

    # convert labels as vector
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY  = lb.fit_transform(testY)
    
    # define 784-256-128-10 architecture using Keras
    model = Sequential()
    model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10,activation="softmax"))

    # train model using SGD
    print("[INFO] training network...")
    sgd = SGD(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, 
                metrics=["accuracy"])
    H = model.fit(trainX, trainY, validation_data=(testX,testY),
                    epochs=100, batch_size=128)

    # evalute the network
    print("[INFO] evaluating network...")
    preds = model.predict(testX)
    print(classification_report(testY.argmax(axis=1),
            preds.argmax(axis=1),
            target_names=[str(x) for x in lb.classes_]))

    # plot training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["output"])

if __name__== '__main__':
    main()

