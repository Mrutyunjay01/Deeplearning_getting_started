# import the necessary packages
from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat
    # construct the preprocessor function

    def preprocess(self, image):
        """
        1. Accepts an image as input
        2. calls img_to_array on the image, ordering the channels
        based on our config file/ value of the dataFormat
        3. Returns a new numpy array with properly ordered channels
        :param image:
        :return:
        """
        return img_to_array(image, data_format=self.dataFormat)
