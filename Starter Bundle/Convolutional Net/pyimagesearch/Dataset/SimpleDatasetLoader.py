# import necessary packages
import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        """
        store the image preprocessor
        :param preprocessors:
        """
        self.preprocessors = preprocessors

        # if preprocessors are none, initialize them as empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        """
        initialize the list of features and labels
        :param imagePaths:
        :param verbose:
        :return:
        """
        data = []
        labels = []

        # looping over the input images
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # /data_set/class/img.jpeg

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] Preprocessed {}/{}".format(i + 1, len(imagePaths)))
        return np.array(data), np.array(labels)