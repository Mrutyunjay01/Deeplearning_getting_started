# import our required packages
import os
import argparse
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from PyimageSearch.NeuralNetworks.ConvulationalNets.minivggnet import MiniVGGNet
# construct our argument parser to save our weights into weights directory
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=True,
                help='Path to our weights direcftory')
args = vars(ap.parse_args())

# loading data
print('[INFO] Loading cifar-10 dataset...')
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# scale the input images pixel vlaues into 0-1 range
train_x = train_x.astype(float)/255.0
test_x = test_x.astype(float)/255.0

# convert the labels from integers to vectors of dimension 10
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

# initialize the optimizer and model
opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01/40)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# construct the call back to save only the best model to the disk based on validation loss
fname = os.path.sep.join([args['weights'],
                          'weights-{epoch:03d}-{val_loss:.4f}.hdf5'])
checkpoint = ModelCheckpoint(fname, monitor='val_loss', verbose=1,
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

# train the network
print('[INFO] Training the network...')
History = model.fit(train_x, train_y,
                    validation_data=(test_x, test_y),
                    batch_size=16,
                    epochs=20,
                    callbacks=callbacks,
                    verbose=2)
