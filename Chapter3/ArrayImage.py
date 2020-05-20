import cv2
import argparse

def option():

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input image")
    ap.add_argument("-n", "--name", type=str, required=False,default="Image")
    args= vars(ap.parse_args())
    return args

def main():
    args = option()
    
    image = cv2.imread(args["dataset"])
    print(args["name"],image.shape)
    """
    cv2.imshow(image)
    cv2.waitKey(0)
    """

if __name__ == '__main__':
    main()

