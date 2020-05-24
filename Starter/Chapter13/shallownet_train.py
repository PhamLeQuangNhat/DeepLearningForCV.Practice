"""use python Chapter13/shallownet_train.py --dataset datasets/animals \
    --output Chapter13/image_animals.png --model Chapter13/shallownet_weights.hdf5""" 

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataloader.simpledatasetloader import SimpleDatasetLoader
from preprocessing.imagetoaraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessing import SimplePreprocessor
from nn.conv.shallownet import ShallowNet

from keras.optimizers import SGD
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse

# construct the argument parser and parse the arguments
def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    ap.add_argument("-o", "--output", required=True,
                help="path to output loss/accuracy plot")
    ap.add_argument("-o", "--model", required=True,
                help="path to output model")
    args = vars(ap.parse_args())
    return args

def main():
    args = option()

    # grab list of images 
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize the image preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()

    # load the dataset from disk
    sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)

    # scale the raw pixel intensities to the range [0, 1]
    data = data.astype("float") / 255.0

    # split training: 75%, testing: 25%
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                    test_size=0.25, random_state=42)

    # convert labels as vector 
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.005)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])
    
    # train the network 
    print("[INFO]training network ...")
    H = model.fit(trainX, trainY, validation_data=(testX,testY),
                batch_size=32, epochs=100, verbose=1)
    
    # save the network to disk 
    print("[INFO] serializing network ...")
    model.save(args["model"])

    # evaluate the network
    print("[INFO] evaluating network...")
    preds = model.predict(testX)
    print(classification_report(testY.argmax(axis=1),
                    preds.argmax(axis=1),
                    target_names=["cat","dog","panda"]))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["output"])

if __name__== '__main__':
    main()


