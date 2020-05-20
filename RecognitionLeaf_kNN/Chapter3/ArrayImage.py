import cv2
import argparse 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input image")
ap.add_argument("-n", "--name", type=str, required=False,default="Image")
args= vars(ap.parse_args())
image = cv2.imread(args["dataset"])
cv2.imshow(args["name"],image)
cv2.waitKey(0)