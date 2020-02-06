# import required packages
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.NeuralNetwork import shallow_net
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# load the training and testing data,
# then scale it into the range [0, 1]
print("[INFO] Loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# convert the label from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# initialize the label names for the CIFAR-10 dataset
labelnames = ["airplane", "automobile", "bird",
              "cat", "deer", "dog", "frog", "horse",
              "ship", "truck"]
# initialize the optimizer and model
print("[INFO] Compiling the Model...")
opt = SGD(lr=0.01)
model = shallow_net.ShallowNet.build(width=32, height=32,
                                     depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])
# train the net
print("[INFO] Training the network...")
H = model.fit(trainX, trainY, batch_size=32,
              epochs=50, verbose=1, validation_data=(testX, testY))
# Evaluate the network
print("[INFO] Evaluating the Net...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelnames))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 50), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and accuracy")
plt.xlabel("# Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.grid()
plt.show()
