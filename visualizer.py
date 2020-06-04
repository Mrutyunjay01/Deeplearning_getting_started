from Utils.NeuralNetworks.ConvulationalNets.lenet import  LeNet
from keras.utils import plot_model

# initialize the lenet model
model = LeNet.build(28, 28, 1, 10)
# visualize the garph and save to disk
plot_model(model, to_file='images/lenet.png',show_shapes=True)
