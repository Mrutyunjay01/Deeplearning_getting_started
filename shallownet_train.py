# import required packages
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.Dataset import SimpleDatasetLoader
from pyimagesearch.NeuralNetwork import shallow_net
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output Model")
args = vars(ap.parse_args())

# Load the images for the dataset
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = simplepreprocessor.SimplePreprocessor(32, 32)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

# load the dataset form the
# scaling the raw pixels into range [0, 1]
sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partition of data test=0.25
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=42)
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
# training the shallow net
print("[INFO] Compiling the Model...")
opt = SGD(lr=0.001)
model = shallow_net.ShallowNet.build(width=32, height=32,
                                     depth=3, classes=3)
model.compile(optimizer=opt, loss="categorical_crossentropy",
              metrics=["accuracy"])
# Train the network
print("[INFO] Training the network")
H = model.fit(trainX, trainY,
              validation_data=(testX, testY), batch_size=32,
              epochs=100, verbose=1)
# Save the trained net to the disk
print("[INFO] Serializing Network...")
model.save(args["model"])
# Evaluate the network
print("[INFO] Evaluating the net...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=["cat", "dog", "panda"]))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="Train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="Val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="Train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="Val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("# Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.grid()
plt.show()
