from sklearn.neighbors import KNeighborsClassifier

# convert labels represented as strings to integers
from sklearn.preprocessing import LabelEncoder

# create training and testing splits
from sklearn.model_selection import train_test_split

# evaluate the performance of classifier and
# print a nicely formatted table of results to console
from sklearn.metrics import classification_report

from dataloader.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import SimplePreprocessor
from imutils import paths

# grab the list of images 
print ("[INFO] loading images ...")
imagePaths = list(paths.list_images("Dataset Leaf"))
# print(len(imagePaths))

# initialize the image preprocessor, load the data set from disk
# and reshape the data matrix
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, label) = sdl.load(imagePaths, verbose=100)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training: 75%, testing: 25%
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
                                 test_size=0,25, random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier ...")
model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model.fit(trainX, trainY)

print(classification_report(testY, model.predict(testX), 
    target_names=le.classes_))

