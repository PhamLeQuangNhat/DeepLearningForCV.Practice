from imutils import paths 
import argparse

# construct the argument parse and parse the arguments

def option():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset",required=True, 
                    help="path to input data")
    ap.add_argument("-k", "--neighbors", type=int, default=1,
                    help="number of neighbors k for classification")
    args = vars(ap.parse_args())
    return args

# main function 
def main():
    args = option()

    imagePaths = list(paths.list_images(args["dataset"]))
    
    print(len(imagePaths))
    

if __name__== '__main__':
    main()
