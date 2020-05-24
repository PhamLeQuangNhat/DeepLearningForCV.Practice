"""use python Chapter13/shallownet_load.py --datatest Chapter13/datatest --model Chapter13/shallownet_weights.hdf5"""

from preprocessing.imagetoaraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessing import SimplePreprocessor
from dataloader.simpledatasetloader import SimpleDatasetLoader

from keras.models import load_model
from imutils import paths
import numpy as np 
import argparse
import cv2

# construct the argument parse and parse the arguments
def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--datatest", required=True,
                        help="path to input datatest")
    ap.add_argument("-m", "--model", required=True,
                    help="path to pre-trained model")
    args = vars(ap.parse_args())
    return args 

def main():
    args = option()

    # initialize class labels 
    classLabels = ["cat","dog","panda"]

    # grab the list of images in the datatest
    print("[INFO] sampling images...")
    imagePaths = np.array(list(paths.list_images(args["datatest"])))

    # initialize the image preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()

    # load the dataset from disk
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths)

    # cale the raw pixel intensities to the range [0, 1]
    data = data.astype("float")/255.0

    # load pre-trained network
    print("[INFO] loading pre-trained network...")
    model = load_model(args["model"])

    # make predictions on the images
    print("[INFO] predicting...")
    preds = model.predict(data).argmax(axis=1)

    # loop over sample images
    for (i, imagePaths) in enumerate(imagePaths):
        # load the example image
        image = cv2.imread(imagePaths)

        # draw the prediction
        cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # save reuslt 
        cv2.imwrite("Chapter13/Image_Predict_{}.png".format(i), image)

if __name__== '__main__':
    main()
