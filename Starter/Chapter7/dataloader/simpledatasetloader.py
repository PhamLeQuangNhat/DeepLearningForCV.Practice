import numpy as np 
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self,preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images 
        for (i, imagePath) in enumerate(imagePaths):
            # load image and extract label
            # format: /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors 
                # apply each to image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)
      
        # show an update images
        if verbose > 0 and i > 0 and (i + 1)% verbose == 0:
            print("[INFO] processed {}/{}".format(i+1,len(imagePaths)))
        
        return (np.array(data),np.array(labels))
