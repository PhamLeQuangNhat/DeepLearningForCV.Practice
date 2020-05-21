""" use python Chapter7/knn.py --dataset datasets/animals"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataloader.simpledatasetloader import SimpleDatasetLoader 
from preprocessing.simplepreprocessing import SimplePreprocessor

from imutils import paths 
import argparse

# construct the argument parse and parse the arguments
def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset",required=True, 
                    help="path to input data")
    ap.add_argument("-k", "--neighbors", type=int, default=1,
                    help="number of neighbors k for classification")
    ap.add_argument("-j","--jobs", type=int, default=-1,
          help="(-1 uses all available cores)")
    args = vars(ap.parse_args())
    return args

# main function 
def main():
    args = option()

    # grab the list of images 
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    
    # initialize image preprocessor
    sp = SimplePreprocessor(32,32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])

    # load dataset from disk
    (data, labels) = sdl.load(imagePaths,verbose=500)

    # reshape the data matrix
    data = data.reshape((data.shape[0],3072))

    # show info on memory consumption of images
    print("[INFO] feature matrix: {:.1f}MB".format(
        data.nbytes/ (1024 *1000)))

    # convert labels from string to vertors
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    #split training:75%, test:25%
    (trainX, testX, trainY, testY) = train_test_split(data,labels,
                                    test_size=0.25, random_state=42)
    
    # train and evaluate a k-NN classifier on the raw pixel intensities
    print("[INFO] evaluating k-NN classifier ...")
    model = KNeighborsClassifier(n_neighbors=args["neighbors"],
            n_jobs=args["jobs"])
    model.fit(trainX,trainY)
    prediction = model.predict(testX)
    print(classification_report(testY, prediction,
          target_names=lb.classes_))

if __name__== '__main__':
    main()

