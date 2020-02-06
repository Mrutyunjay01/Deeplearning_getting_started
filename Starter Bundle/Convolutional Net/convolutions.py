# import the necessary packages
import numpy as np
from skimage.exposure import rescale_intensity
import argparse
import cv2


def convolve(image, K):
    # grab the spatial dimensions of the image and kernel
    (I_height, I_width) = image.shape[:2]
    (K_height, K_width) = K.shape[:2]

    # allocate memory for the output image, taking care to padding
    # the borders of the input image so the spatial size (i.e Width and height)
    # are not reduced
    pad = (K_width - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((I_height, I_width), dtype="float")

    # loop over the input image, "sliding" the kernel across
    # each (x, y) coordinate from left to right and top to bottom
    for y in np.arange(pad, I_height + pad):
        for x in np.arange(pad, I_width + pad):
            # extract the roi of the image by extracting the center region
            # of the current x, y coordinate dimensions
            roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1]
            # perform the actual convolution by taking the element-wise
            # multiplication between the ROI and the kernel, Then summing the matrix
            C = (roi * K).sum()

            # store the convolved vlaue in the output (x, y)
            # co-ordinate of the output image
            output[y - pad, x - pad] = C
            # rescale the output image to be in the range of (0, 255)
            output = rescale_intensity(output, in_range=(0, 255))
            output = (output * 255).astype("uint8")

            # return the output image
            return output
# construct the argument parse the parse the arguments


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="Path to the input image")
args = vars(ap.parse_args())
# construct average burring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
# sharpening filter
sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype="int")
# Laplacian kernel for edge detection
laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype="int")
# Sobel kernels : The Sobel kernels can be used to detect edge-like regions
# along both the x and y axis, respectively:
sobelX = np.array(([-1, 0, 1],
                   [-2, 0, 1],
                   [-1, 0, 1]), dtype="int")
sobelY = np.array(([-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]), dtype="int")
# emboss kernel
emboss = np.array(([-2, -1, 0],
                  [-1, 1, 1],
                  [0, 1, 2]), dtype="int")
# construct the kernel bank, a list of kernels we’re going to apply
# using both our custom ‘convole‘ function and OpenCV’s ‘filter2D‘
# function
kernelBank = (("small_blur", smallBlur),
              ("large_blur", largeBlur),
              ("sharpen", sharpen),
              ("laplacian", laplacian),
              ("sobel_x", sobelX),
              ("sobel_y", sobelY),
              ("emboss", emboss))
# load the input image and convert it to grayscale
image = cv2.imread(args["input"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# loop over the kernels
for (kernelName, k) in kernelBank:
    # apply the kernel to the grayscale image using
    # both our custom 'convolve' functions and OpenCV's Fucn
    print("[INFO] Applying {} kernel".format(kernelName))
    convoOutput = convolve(gray, k)
    opencvOutput = cv2.filter2D(gray, -1, k)

    # show the ouput images
    cv2.imshow("Original", image)
    cv2.imshow("Grayscaled", gray)
    cv2.imshow("{} - convolve".format(kernelName), convoOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()