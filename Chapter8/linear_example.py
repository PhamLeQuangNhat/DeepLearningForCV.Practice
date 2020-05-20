import cv2
import numpy as np
import argparse

def option():

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input image")
    ap.add_argument("-n", "--name", type=str, required=False,default="Image")
    ap.add_argument("-o", "--output", required=True, help="path to oput image")
    args= vars(ap.parse_args())
    return args

def main():
    args = option()
    #print(args)
    
    # initialize class labels and set the seed of the pseudoradom
    # number generator so we can reproduce our results
    labels = ["dog","cat","panda"]
    np.random.seed(1)

    # randomly initialize our weight matrix and bias vector
    W = np.random.randn(3,3072)
    b = np.random.randn(3)

    # load image, flatten it 
    orig_image = cv2.imread(args["dataset"])
    image = cv2.resize(orig_image,(32,32)).flatten()

    # compute output score 
    scores = W.dot(image) + b
    
    # loop over the scores + labels and display them
    for (label, score) in zip(labels, scores):
        print("[INFO] {}: {:.2f}".format(label, score))
    
    # draw the label with the highest score on the image
    # as our prediction
    org_image= cv2.putText(orig_image, "Label: {}".format(labels[np.argmax(scores)]),
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # save result 
    cv2.imwrite(args["output"],org_image)
    
if __name__ == '__main__':
    main()

