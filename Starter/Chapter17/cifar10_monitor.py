"""use python3 Chapter15/minivggnet_cifar10.py --dataset datasets/cifar10/ --output Chapter15/image_cifar10.png"""
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from callbacks.trainingmonitor import TrainingMonitor
from nn.conv.minivggnet import MiniVGGNet
from keras import backend as K 
import argparse
from keras.optimizers import SGD

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
                    help="path to output directory")
    args = vars(ap.parse_args())
    return args

def main():
    args = option()

    # show information on the process ID
    print("[INFO process ID: {}".format(os.getpid()))

    # load training data and testing data 
    trainX, trainY = load_data_train(args["dataset"])
    testX, testY = load_data_test(args["dataset"])
    
    # reshape data matrix
    if K.image_data_format() == "channels_first":
        trainX = trainX.reshape(trainX.shape[0],3,32,32)
        testX = testX.reshape(testX.shape[0],3,32,32)
    else:
        trainX = trainX.reshape(trainX.shape[0],32,32,3)
        testX = testX.reshape(testX.shape[0],32,32,3)
    
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

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])
    
    # construct set of callbacks
    figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])

    jsonPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
    callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

    # train the network
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                            batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
                        
    # evaluate network
    print("[INFO] evaluating network...")
    preds = model.predict(testX)
    print(classification_report(testY.argmax(axis=1),
                preds.argmax(axis=1), target_names=labelNames))

if __name__== '__main__':
    main()