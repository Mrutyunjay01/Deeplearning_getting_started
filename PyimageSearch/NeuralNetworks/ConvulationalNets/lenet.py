# import required packages
from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
# Build the constructor called LeNet


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channel first" , update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # Now set the first layer of the neural Net
        # CONV => RELU => POOL LAYERS
        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=inputShape))
        # 20 kernels of size 5 * 5 and same padding
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Apply our second layer of neural net
        # CONV => RELU => POOL Layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Now apply our 3rd layer of our neural net
        # A fully connected layer with 500 weights
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Now apply our last and classification layer called Softmax layer
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the Model
        return model





