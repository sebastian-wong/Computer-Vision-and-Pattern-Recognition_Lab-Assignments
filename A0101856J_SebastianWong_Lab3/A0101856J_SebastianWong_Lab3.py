import numpy as numpy
import cv2
from matplotlib import pyplot as plt


# Detecting vertical edges
sobelHorizontal = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
prewittHorizontal = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
# Detecting horizontal edges
sobelVertical = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
prewittVertical = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
img = cv2.imread(os.getcwd() + "/pictures/test1.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
imgShape = img.shape()
imgWidth = imgShape[0]
imgHeight = imgShape[1]
# Filtering
# for every pixel
for x in range(0, imgWidth-2)
	for y in range(0, imgHeight-2)
	# each matrix placement
	pixelX = (sobelHorizontal[0][0] * img[x][y]) + (sobelHorizontal[0][1] * img[x+1][y]) + (sobelHorizontal[0][2] * img[x+2][y])
	+ ()

