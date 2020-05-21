import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and 
        # interpolation method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
    
    def preprocess(self, image):
        # ignore aspect ratio
        # resize image to a fixed size
        return cv2.resize(image, (self.width, self.height),
                        interpolation=self.inter)
