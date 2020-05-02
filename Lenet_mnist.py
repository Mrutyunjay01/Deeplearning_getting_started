# import required packages
from sklearn.datasets import fetch_openml
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from PyimageSearch.NeuralNetworks.ConvulationalNets import lenet
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
# Load the dataset from the disk
print("[INFO] Accessing MNIST full dataset")
dataset = fetch_openml('mnist_784', version=1)
data = dataset.data

# for confirmation purpose, "channels_first" check up is needed
if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)
# otherwise, we are using "channels_last" ordering,
# keep the design as it should be
else:
    data = data.reshape(data.shape[0], 28, 28, 1)
# Now as we are done with our model loading,
# lets move to out model designing and get done with input scaling
(trainX, testX, trainY, testY) = train_test_split(data / 255.0,
                                                  dataset.target.astype("int"), test_size=0.25,
                                                  random_state=42)
# convert the labels from integers to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize the optimizer and our main Model
print("[INFO] Compiling Model...")
opt = SGD(lr=0.01)
model = lenet.LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=['accuracy'])
# train the network
print("[INFO] Training the Net...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=128, epochs=20, verbose=1)
# Evaluate the network
print("[INFO] Evaluating the network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))
# plot the training loss and accuracy

plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, 20), H.history["loss"], label="Training_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="Val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="Training_accuracy")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="Validation_accuracy")
plt.title("Training Loss and accuracy")
plt.xlabel("# Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
