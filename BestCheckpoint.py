# import required packages
import os
import argparse
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from PyimageSearch.NeuralNetworks.ConvulationalNets.minivggnet import MiniVGGNet

# create argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=True,
                help='path to the best weights')
args = vars(ap.parse_args())

# loading our dataset
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# scale our input pixels into (0, 1) range
train_x = train_x.astype(float) / 255.0
test_x = test_x.astype(float) / 255.0

# convert our output labels into vectors of dimension same as classes
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

# initialize the optimizer and model
opt = SGD(learning_rate=0.01, decay=0.01 / 40,
          momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# construct the callback to save our best model
checkpoint = ModelCheckpoint(args['weights'],
                             monitor='val_loss',
                             save_best_only=True,
                             verbose=2)
callbacks = [checkpoint]
"""
the fname template string is gone – all we are doing is supply the value of
--weights to ModelCheckpoint. Since there are no template values to fill in, Keras will simply
overwrite the existing serialized weights file whenever our monitoring metric improves (in this
case, validation loss).
Finally, we train on the network below.
"""

print('[INFO] Train the network..')
history = model.fit(train_x, train_y,
                    validation_data=(test_x, test_x),
                    epochs=40,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=2)
# saves the best weight file to given path
# python BestCheckpoint.py --weights Weighs/cifar10Best
