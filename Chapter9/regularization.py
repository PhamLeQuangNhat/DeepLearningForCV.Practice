""" use python Chapter9/regularization.py --dataset datasets/animal """

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from dataloader.simpledatasetloader import SimpleDatasetLoader 
from preprocessing.simplepreprocessing import SimplePreprocessor

from imutils import paths 
import argparse

def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset",required=True, 
                    help="path to input data")    
    args = vars(ap.parse_args())
    return args

def main():
    args = option()
    
    # grab the list of image path
    print("[INFO] loading images ...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize the image preprocessor
    sp = SimplePreprocessor(32, 32)
    
    # load dataset from disk
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=500)

    # reshape th data matrix
    data = data.reshape((data.shape[0], 3072))

    # encode labels as vector
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # split training:75%, test:25%
    (trainX, testX, trainY, testY) = train_test_split(data,labels,
                                    test_size=0.25, random_state=42)
    
    # loop over set of regularizers
    for r in (None, "l1", "l2"):
	# train a SGD classifier using a softmax loss function and
        # specified regularization function for 10 epochs
        print("[INFO] training model with ‘{}‘ penalty".format(r))
	model = SGDClassifier(loss="log", penalty=r, max_iter=10,
	         learning_rate="constant", eta0=0.01, random_state=42)
        model.fit(trainX, trainY)

	# evalute classifier
	acc = model.score(testX,tesxtY)
	print("[INFO] ‘{}‘ penalty accuracy: {:.2f}%".format(r,acc * 100))

if __name__ == '__main__':
    main()
