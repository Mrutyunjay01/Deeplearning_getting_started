import matplotlib
import argparse
from keras.datasets import cifar10
from PyimageSearch.NeuralNetworks.ConvulationalNets import minivggnet

matplotlib.use('Agg')
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# define our custom learning rate scheduler
def step_decay(epoch):
    """

    :param epoch: epoch of our training schedule
    :return: decayed learning rate
    """
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    # computer our function
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    return float(alpha)


'''
# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help='path to output plot')
args = vars(ap.parse_args())
'''
# load the training and testing data
print('[INFO] Loading CIFAR-10 data...')
((train_X, train_Y), (test_X, test_Y)) = cifar10.load_data()

# scale them into (0, 1) range
train_X = train_X.astype(float) / 255.0
test_X = test_X.astype(float) / 255.0

# converts the labels from integers to vectors
lb = LabelBinarizer()
train_Y = lb.fit_transform(train_Y)
test_Y = lb.fit_transform(test_Y)

# initialize the label names
labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

# define the set of callbacks
callback = [LearningRateScheduler(step_decay)]

# initialize the optimizer and the model
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = minivggnet.MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# train the network and save the history
H = model.fit(train_X, train_Y, validation_data=(test_X, test_Y),
              batch_size=64, epochs=40, callbacks=callback, verbose=1)

# evaluate the network
print('[INFO] Evaluating the net..')
predictions = model.predict(test_X, batch_size=64)
print(classification_report(test_Y.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=labelNames))

# plot our training loss and accuracy
plt.style.use('gg')
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 40), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 40), H.history['accuracy'], label='train_accuracy')
plt.plot(np.arange(0, 40), H.history['val_accuracy'], label='val_accuracy')
plt.title('Training loss/accuracy on CIFAR-10')
plt.xlabel('#Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(r'C:\Users\MRUTYUNJAY BISWAL\PycharmProjects\images')
