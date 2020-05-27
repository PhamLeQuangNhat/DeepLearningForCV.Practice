from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from nn.conv.minivggnet import MiniVGGNet
from preprocessing.imagetoaraypreprocessor import ImageToArrayPreprocessor
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from dataloader.simpledatasetloader import SimpleDatasetLoader

from keras.optimizers import SGD
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
    #ap.add_argument("-o", "--output", required=True,
    #                help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())
    return args

def main():
    args = option()
    # grab the list of images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    #print(len(imagePaths))
    classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
    classNames = [str(x) for x in np.unique(classNames)]
    #print(len(classNames))
    

if __name__=='__main__':
    main()    