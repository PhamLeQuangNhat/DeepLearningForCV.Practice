import argparse
import os
import numpy as np
from imutils import paths
import cv2
imagePaths = list(paths.list_images('datasets/animals/'))
print(len(imagePaths))
data = []
labels = []

# loop over the input images 
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-2]
    data.append(image)
    labels.append(label)
      


