"""
Create a keras callback to be calles at the end of every epoch
and babysit our training and validation loss.
"""
# import necessary packages
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        """

        :param figPath: path to output the plot
        :param jsoPath: serialize the loss and accuracy over time, useful in training history
        :param startAt: starting epoch
        """
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the josn path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check whether the starting point of training is supplied

                if self.startAt > 0:
                    # loop over the entire log and trim any entries that are past training epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy etc for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        if self.jsonPath is not None:
            f = open(self.jsonPath, 'w')
            f.write(json.dumps(str(self.H)))
            f.close()

        if len(self.H['loss']) > 1:
            N = np.arange(0, len(self.H['loss']))
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['loss'], label='train_loss')
            plt.plot(N, self.H['val_loss'], label='val_loss')
            plt.plot(N, self.H['accuracy'], label='train_accuracy')
            plt.plot(N, self.H['val_accuracy'], label='val_accuracy')
            plt.title('Training loss and Accuracy [Epoch {}]'.format(len(self.H['loss'])))
            plt.xlabel('# Epoch')
            plt.ylabel('Loss/ Accuracy')
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()

