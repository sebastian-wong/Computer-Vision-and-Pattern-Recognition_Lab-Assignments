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

cv2.imwrite("grey.jpg", img)	

imgShape = img.shape
imgHeight = imgShape[0]
imgWidth = imgShape[1]
sobelResults = np.zeros([imgHeight,imgWidth])
prewittResults = np.zeros([imgHeight,imgWidth])
maxEdgeStrSobel = maxEdgeStrPrewitt = 0
maxHorizontalEdgeStrSobel = maxVerticalEdgeStrSobel = 0
maxHorizontalEdgeStrPrewitt = maxVerticalEdgeStrPrewitt = 0

# Filtering
# for every pixel
for x in range(0, imgHeight-2):
	for y in range(0, imgWidth-2):
		# each matrix placement
		pixelXSobel = pixelYSobel = pixelXPreWitt = pixelYPreWitt = edgeStrSobel = edgeStrPewitt = 0

		pixelXSobel = ((sobelHorizontal[0][0] * img[x][y]) + (sobelHorizontal[0][1] * img[x][y+1]) + (sobelHorizontal[0][2] * img[x][y+2]) + 
		(sobelHorizontal[1][0] * img [x+1][y]) + (sobelHorizontal[1][1] * img[x+1][y+1]) + (sobelHorizontal[1][2] * img[x+1][y+2]) + 
		(sobelHorizontal[2][0] * img[x+2][y]) + (sobelHorizontal[2][1] * img[x+2][y+1]) + (sobelHorizontal[2][2] * img[x+2][y+2]))

		pixelYSobel = ((sobelVertical[0][0] * img[x][y]) + (sobelVertical[0][1] * img[x][y+1]) + (sobelVertical[0][2] * img[x][y+2]) + 
		(sobelVertical[1][0] * img [x+1][y]) + (sobelVertical[1][1] * img[x+1][y+1]) + (sobelVertical[1][2] * img[x+1][y+2]) + 
		(sobelVertical[2][0] * img[x+2][y]) + (sobelVertical[2][1] * img[x+2][y+1]) + (sobelVertical[2][2] * img[x+2][y+2]))

		pixelXPreWitt = ((prewittHorizontal[0][0] * img[x][y]) + (prewittHorizontal[0][1] * img[x][y+1]) + (prewittHorizontal[0][2] * img[x][y+2]) + 
		(prewittHorizontal[1][0] * img [x+1][y]) + (prewittHorizontal[1][1] * img[x+1][y+1]) + (prewittHorizontal[1][2] * img[x+1][y+2]) + 
		(prewittHorizontal[2][0] * img[x+2][y]) + (prewittHorizontal[2][1] * img[x+2][y+1]) + (prewittHorizontal[2][2] * img[x+2][y+2]))

		pixelYPreWitt = ((prewittVertical[0][0] * img[x][y]) + (prewittVertical[0][1] * img[x][y+1]) + (prewittVertical[0][2] * img[x][y+2]) + 
		(prewittVertical[1][0] * img [x+1][y]) + (prewittVertical[1][1] * img[x+1][y+1]) + (prewittVertical[1][2] * img[x+1][y+2]) + 
		(prewittVertical[2][0] * img[x+2][y]) + (prewittVertical[2][1] * img[x+2][y+1]) + (prewittVertical[2][2] * img[x+2][y+2]))
		# Calculating edge strength	
		edgeStrSobel = math.sqrt((math.pow(abs(pixelXSobel),2)) + ((math.pow(abs(pixelYSobel),2))))
		edgeStrPewitt = math.sqrt((math.pow(abs(pixelXPreWitt),2)) + ((math.pow(abs(pixelYPreWitt),2))))
		# Finding max edge strength
		if (maxEdgeStrSobel < edgeStrSobel):
			maxEdgeStrSobel = edgeStrSobel	
		# Finding max edge strength
		if (maxEdgeStrPrewitt < edgeStrPewitt):
			maxEdgeStrPrewitt = edgeStrPewitt

		sobelResults[x+1][y+1] = edgeStrSobel
		prewittResults[x+1][y+1] = edgeStrPewitt

cv2.imwrite("example_sobel.jpg", sobelResults)
cv2.imwrite("example_prewitt.jpg", prewittResults)

# Scaling results to 255
sobelResults = sobelResults * (255/maxEdgeStrSobel)
prewittResults = prewittResults * (255/maxEdgeStrPrewitt)		
cv2.imwrite("example_sobel_scaled.jpg", sobelResults)
cv2.imwrite("example_prewitt_scaled.jpg", prewittResults)

#thinning of Sobel and Prewitt edges
maxHorizontalEdgeStrSobel = maxVerticalEdgeStrSobel = 0
maxHorizontalEdgeStrPrewitt = maxVerticalEdgeStrPrewitt = 0
r , c = sobelResults.shape
sobelResultsThinned = prewittResultsThinned = np.zeros([r,c])
for rows in range(len(sobelResults)):
	maxHorizontalEdgeStrSobel = np.max(sobelResults[rows,:])
	maxHorizontalEdgeStrPrewitt = np.max(prewittResults[rows,:])
	for columns in range(len(sobelResults[rows])):
		maxVerticalEdgeStrSobel = np.max(sobelResults[:,columns])
		maxVerticalEdgeStrPrewitt = np.max(prewittResults[:,columns])
		if ((sobelResults[rows][columns] >= maxHorizontalEdgeStrSobel-20) or (sobelResults[rows][columns] >= maxVerticalEdgeStrSobel-20)):
			sobelResultsThinned[rows][columns] += sobelResults[rows][columns]
		if ((prewittResults[rows][columns] >= maxHorizontalEdgeStrPrewitt-20) or (prewittResults[rows][columns] >= maxVerticalEdgeStrPrewitt-20)):
			prewittResultsThinned[rows][columns] += prewittResults[rows][columns]
		maxVerticalEdgeStrSobel = maxVerticalEdgeStrPrewitt = 0
	maxHorizontalEdgeStrSobel = maxHorizontalEdgeStrPrewitt = 0			
cv2.imwrite("example_sobel_scaled_thinned.jpg", sobelResultsThinned)
cv2.imwrite("example_prewitt_scaled_thinned.jpg", prewittResultsThinned)			
				 

