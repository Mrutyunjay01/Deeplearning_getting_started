# import required packages
import argparse
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from PyimageSearch.NeuralNetworks.ConvulationalNets import minivggnet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
# construct an argument parser for saving our graphs
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load the training and testing data
# and scale them into range of [0, 1]
print("[INFO] Loading the CIFAR-10 Data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# initialize the label names for CIFAR-10 data set
labelNames = ['airplane', 'automobile', 'bird', 'cat',
              'deer', 'frog', 'horse', 'ship', 'truck']
# compiling our model
print("[INFO] Compiling our model...")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = minivggnet.MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# train the net
print("[INFO] Training the Net...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64, epochs=40, verbose=1)
# decay parameter is used to slowly reduce the learning rate over time
# Evaluate our network
print("[INFO] Evaluating our network")
predictions = model.predict(testY, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=labelNames))
# plot the observation
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="training_accuracy")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and accuracy on CIFAR-10")
plt.xlabel("#Epochs")
plt.ylabel("loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
