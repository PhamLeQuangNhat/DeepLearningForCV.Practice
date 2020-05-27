"""use python3 Chapter2/minivgg_flowers17.py --dataset datasets/flowers17/ --output Chapter2/image_aug.png"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from nn.conv.minivggnet import MiniVGGNet
from preprocessing.imagetoaraypreprocessor import ImageToArrayPreprocessor
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from dataloader.simpledatasetloader import SimpleDatasetLoader

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import os 

# construct the argument parse and parse the arguments
def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())
    return args

def main():
    args = option()

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest")

    # grab the list of images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    
    classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
    classNames = [str(x) for x in np.unique(classNames)]
    
    # initialize the image preprocessors
    aap = AspectAwarePreprocessor(64, 64)
    iap = ImageToArrayPreprocessor()

    # load the dataset from disk
    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
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
    opt = SGD(lr=0.05)
    model = MiniVGGNet.build(width=64, height=64, depth=3,
                                classes=len(classNames))
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                    metrics=["accuracy"])
    
    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
        epochs=100, verbose=1)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=classNames))

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

if __name__=='__main__':
    main()    