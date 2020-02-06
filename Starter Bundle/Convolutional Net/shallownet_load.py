# import required packages
import argparse
import numpy as np
from imutils import paths
from pyimagesearch.preprocessing import simplepreprocessor, imagetoarraypreprocessor
from pyimagesearch.Dataset import SimpleDatasetLoader
from keras.models import load_model
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())
# initialize class labels
classLabels = ["cat", "dog", "panda"]
# Loading the dataset from the disk
# Randomly sample indexes into the image path list
print("[INFO] Sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
indexes = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[indexes]

# initialize the image preprocessors
sp = simplepreprocessor.SimplePreprocessor(32, 32)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()
# Load the dataset from the disk and scale into [0, 1]
sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# Load the pre-trained network
print("[INFO] Loading the pre-trained Net...")
model = load_model(args["model"])
# make predictions on images
print("[INFO] Predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)
# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
