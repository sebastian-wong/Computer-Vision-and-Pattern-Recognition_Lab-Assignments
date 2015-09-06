import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
# Working directory
# E:/My Documents/GitHub/Computer-Vision-and-Pattern-Recognition_Lab-Assignments/A0101856J_SebastianWong_Lab3
##http://pythongeek.blogspot.sg/2012/06/canny-edge-detection.html#comment-form
##http://docs.gimp.org/en/plug-in-convmatrix.html
## https://en.wikipedia.org/wiki/Kernel_(image_processing)

# Detecting vertical edges
sobelHorizontal = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
prewittHorizontal = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
# Detecting horizontal edges
sobelVertical = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
prewittVertical = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

img = cv2.imread(os.getcwd() + "/pictures/example.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
imgShape = img.shape
imgHeight = imgShape[0]
imgWidth = imgShape[1]
sobelResults = np.zeros([imgHeight,imgWidth])
prewittResults = np.zeros([imgHeight,imgWidth])

# Filtering
# for every pixel
for x in range(0, imgHeight-2):
	for y in range(0, imgWidth-2):
		# each matrix placement
		pixelXSobel = pixelYSobel = pixelXPreWitt = pixelYPreWitt = edgeStrSobel = edgeStrPewitt = 0

		pixelXSobel = (sobelHorizontal[0][0] * img[x][y]) + (sobelHorizontal[0][1] * img[x][y+1]) + (sobelHorizontal[0][2] * img[x][y+2])
		+ (sobelHorizontal[1][0] * img [x+1][y]) + (sobelHorizontal[1][1] * img[x+1][y+1]) + (sobelHorizontal[1][2] * img[x+1][y+2])
		+ (sobelHorizontal[2][0] * img[x+2][y]) + (sobelHorizontal[2][1] * img[x+2][y+1]) + (sobelHorizontal[2][2] + img[x+2][y+2])

		pixelYSobel = (sobelVertical[0][0] * img[x][y]) + (sobelVertical[0][1] * img[x][y+1]) + (sobelVertical[0][2] * img[x][y+2])
		+ (sobelVertical[1][0] * img [x+1][y]) + (sobelVertical[1][1] * img[x+1][y+1]) + (sobelVertical[1][2] * img[x+1][y+2])
		+ (sobelVertical[2][0] * img[x+2][y]) + (sobelVertical[2][1] * img[x+2][y+1]) + (sobelVertical[2][2] + img[x+2][y+2])

		pixelXPreWitt = (prewittHorizontal[0][0] * img[x][y]) + (prewittHorizontal[0][1] * img[x][y+1]) + (prewittHorizontal[0][2] * img[x][y+2])
		+ (prewittHorizontal[1][0] * img [x+1][y]) + (prewittHorizontal[1][1] * img[x+1][y+1]) + (prewittHorizontal[1][2] * img[x+1][y+2])
		+ (prewittHorizontal[2][0] * img[x+2][y]) + (prewittHorizontal[2][1] * img[x+2][y+1]) + (prewittHorizontal[2][2] + img[x+2][y+2])

		pixelYPreWitt = (prewittVertical[0][0] * img[x][y]) + (prewittVertical[0][1] * img[x][y+1]) + (prewittVertical[0][2] * img[x][y+2])
		+ (prewittVertical[1][0] * img [x+1][y]) + (prewittVertical[1][1] * img[x+1][y+1]) + (prewittVertical[1][2] * img[x+1][y+2])
		+ (prewittVertical[2][0] * img[x+2][y]) + (prewittVertical[2][1] * img[x+2][y+1]) + (prewittVertical[2][2] + img[x+2][y+2])

		#edgeStrSobel = math.sqrt((math.pow(abs(pixelXSobel),2)) + ((math.pow(abs(pixelYSobel),2))))
		#edgeStrPewitt = math.sqrt((math.pow(abs(pixelXPreWitt),2)) + ((math.pow(abs(pixelYPreWitt),2))))


		edgeStrSobel = math.sqrt((pixelXSobel**2) + (pixelYSobel**2))
		edgeStrPewitt = math.sqrt((pixelXPreWitt**2) + (pixelYPreWitt**2))
		sobelResults[x+1][y+1] = edgeStrSobel
		prewittResults[x+1][y+1] = edgeStrPewitt

cv2.imwrite("examplesobel.jpg", sobelResults)
cv2.imwrite("exampleprewitt.jpg", prewittResults)

#thinning of sobel image
thinningC, thinningR = sobelResults.shape
img_thinned = np.zeros((thinningC, thinningR))
prevIntensity = 0
curIntensity = 0
for thinningIndexR in range(0, thinningR):
    for thinningIndexC in range(0, thinningC):
        if(thinningIndexC < thinningC-2 and thinningIndexR < thinningR-2):
            prevIntensity = sobelResults[thinningIndexC, thinningIndexR]
            curIntensity = sobelResults[thinningIndexC+1, thinningIndexR]
            if(curIntensity + 15 < prevIntensity):
                img_thinned[thinningIndexC, thinningIndexR] = prevIntensity
cv2.imwrite('sobel_thinned.jpg',img_thinned)