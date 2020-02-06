# import the necessary packages
from keras.models import Sequential
from keras import backend as k
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

# Construct a class for ShallowNet


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        """

        :param width: Width of the input image that will be used to train the network
        :param height: The height of the input image i.e rows of the array
        :param depth: The number of channels in the input image
        :param classes: Total no of classes our network should learn to predict
        :return: Model
        """
        model = Sequential()
        inputShape = (height, width, depth)  # initialize assuming "Channel last" data format
        # if "Channel_first", update the input shape
        if k.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the 2st layer
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        # add the softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
