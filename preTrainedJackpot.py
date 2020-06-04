# import the nets and utils
from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import argparse
import cv2

# argument parser to prompt the image file and and which net to use
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to input the image')
ap.add_argument('-m', '--model', required=True,
                help='Which network to use')
args = vars(ap.parse_args())

# define a mapping of model names to string
MODELS = {
    'vgg16': VGG16,
    'vgg19': VGG19,
    'inception': InceptionV3,
    'xception': Xception,
    'resnet': ResNet50
}

# sanity check on prompting a valid model according to dictionary
if args['model'] not in MODELS.keys():
    raise AssertionError('Model not present in MODELS dict, update the list')

# initialize the input shape for VGG and resnet : 224*224
input_shape = (224, 224)
# import preprocess_input for inception and Xception layer
from keras.applications.inception_v3 import preprocess_input
preprocess = imagenet_utils.preprocess_input # for rest of the networks

if args['model'] in ('inception', 'xception'):
    input_shape = (229, 229) # these nets accept this shape
    preprocess = preprocess_input

# load the network weights
print('Loading weights...{}'.format(args['model']))
Network = MODELS[args['model']]
model = Network(weights='imagenet')

# load the input image via keras preprocessing
print('Loading and preprocessing image...')
image = load_img(args['image'], target_size=input_shape)
# convert to array
image = img_to_array(image)

# to pass it through the net, expand the dimension to add the no of input
image = np.expand_dims(image)

# perform preprocessing
image = preprocess(image)

# classify the image
print('Classifying the image with {} model...'.format(args['model']))
prediction = model.predict(image)
label = imagenet_utils.decode_predictions(prediction)  # extract the label name

# loop over the predictions and display the top-5 predicitons with max prob
for (i, (imagnetId, lb, prob)) in enumerate(label[0]):
    print("{}. {}: {:.2f}%".format(i+1, lb, prob*100))

# visualize the results :P
original_image = cv2.imread(args['image'])
(imagenetId, lb, prob) = label[0][0]
cv2.putText(original_image,
            'Label : {}'.format(lb),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)
cv2.imshow('Classified as : ', original_image)
cv2.waitKey(0)
