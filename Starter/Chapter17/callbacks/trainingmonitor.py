from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt 
import numpy as np 
import json 
import os 

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store output path for figure, path to the JSON 
        # serialized file, starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        # initialize history dictionary 
        self.H = {}

        # if JSON history path exists, load training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over entries in history log and
                    # trim any entries that are past starting epoch 
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
    
    def on_epoch_end(self, epoch, logs={}):
        # loop over logs and update loss, accuracy
        # for entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        
        # check to see if training history should be
        # serialized to file 
        if self.jsonPath is not None:
            f  = open(self.jsonPath, "w")
            f.write(json.dumps(str(self.H)))
            f.close()

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
                                        len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # save the figure
            plt.savefig(self.figPath)
            plt.close()
