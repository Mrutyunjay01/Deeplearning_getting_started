#import required packages
import os
import matplotlib
import argparse
from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from Utils.callbacks.trainingMonitor import TrainingMonitor
from Utils.NeuralNetworks.ConvulationalNets.minivggnet import MiniVGGNet

print('[INFO] Process id: {}'.format(os.getpid()))
# creating the output parser
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--outputPath', required=True,
                help='Path to the output directory')
args = vars(ap.parse_args())
# load the dataset
print('[INFO] Loading the dataset...')
(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

# scale the images into range(0, 1)
train_X = train_X.astype(float)/255.0
test_X = test_X.astype(float)/255.0

# convert the labels into vectors
lb = LabelBinarizer()
train_Y = lb.fit_transform(train_Y)
test_Y = lb.fit_transform(test_Y)

# load the label names
labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

print('[INFO] Compiling the model')
opt = SGD(momentum=0.9, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# construct the set of callbacks
figPath = os.path.join(args['outputPath'], "{}.png".format(os.getpid()))
jsonPath = os.path.sep.join([args['outputPath'], "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath)]

# train the network
print("[INFO] Training the net...")
model.fit(train_X, train_Y,
          validation_data=(test_X, test_Y),
          batch_size=64, epochs=40,
          callbacks=callbacks, verbose=1)
