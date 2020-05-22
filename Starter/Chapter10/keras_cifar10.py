"""use python3 Chapter10/keras_cifar10.py --dataset datasets/cifar10/ --output Chapter10/image_cifar10.png"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

import argparse
import matplotlib.pyplot as plt 

import pickle
import os
import numpy as np

# read file 
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_train(path):

    # load data from file data_batch_1
    data_dict_1 = unpickle(os.path.join(path, 'data_batch_1'))
    trainX = data_dict_1[b'data']
    trainY = data_dict_1[b'labels']
    
    for i in range (2,6):
        paths = os.path.join(path, 'data_batch_%s' % i)
        data_dict = unpickle(paths)
        tr_data = np.array(data_dict[b'data'])
        tr_labels = np.array(data_dict[b'labels'])
        
        # concatenate 2 array
        trainX = np.concatenate((trainX,tr_data), axis=0)
        trainY = np.concatenate((trainY,tr_labels), axis=0)
        
    return (np.array(trainX), np.array(trainY))

def load_data_test(path):

    # load data test from test_batch
    data_dict = unpickle(os.path.join(path, 'test_batch'))
    testX = data_dict[b'data']
    testY = data_dict[b'labels']
    return (np.array(testX), np.array(testY))

# construct argument parse and parse arguments 
def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to dataset")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())
    return args

def main():
    args = option()

    # load training data and testing data 
    trainX, trainY = load_data_train(args["dataset"])
    testX, testY = load_data_test(args["dataset"])

    # scale trainX, testX into range [0,1]
    trainX = trainX.astype("float") / 255.0
    testX = testX.astype("float") / 255.0

    # convert labels as vector 
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY  = lb.fit_transform(testY)

    # initialize the label names for the CIFAR-10 dataset
    labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]

    # define 3072-1024-512-10 architecture using Keras
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # train model using SGD
    print("[INFO] training network...")
    model.compile(loss="categorical_crossentropy", optimizer=sgd, 
                metrics=["accuracy"])
    H = model.fit(trainX, trainY, validation_data=(testX,testY),
                    epochs=100, batch_size=32)
    
    # evaluate network
    print("[INFO] evaluating network...")
    preds = model.predict(testX)
    print(classification_report(testY.argmax(axis=1),
                preds.argmax(axis=1), target_names=labelNames))

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
    
