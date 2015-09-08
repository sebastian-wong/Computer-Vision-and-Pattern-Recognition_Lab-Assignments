import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
# Working directory

# Detecting vertical edges
sobelHorizontal = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
prewittHorizontal = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
# Detecting horizontal edges
sobelVertical = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
prewittVertical = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

img = cv2.imread(os.getcwd() + "/pictures/test3.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)	
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
		pixelXSobel = pixelYSobel = pixelXPreWitt = pixelYPreWitt = edgeStrSobel = edgeStrPrewitt = 0

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
		edgeStrPrewitt = math.sqrt((math.pow(abs(pixelXPreWitt),2)) + ((math.pow(abs(pixelYPreWitt),2))))
		# Finding max edge strength
		if (maxEdgeStrSobel < edgeStrSobel):
			maxEdgeStrSobel = edgeStrSobel	
		# Finding max edge strength
		if (maxEdgeStrPrewitt < edgeStrPrewitt):
			maxEdgeStrPrewitt = edgeStrPrewitt

		sobelResults[x+1][y+1] = edgeStrSobel
		prewittResults[x+1][y+1] = edgeStrPrewitt

cv2.imwrite("test3_sobel.jpg", sobelResults)
cv2.imwrite("test3_prewitt.jpg", prewittResults)

# Scaling results to 255
sobelResults = sobelResults * (255/maxEdgeStrSobel)
prewittResults = prewittResults * (255/maxEdgeStrPrewitt)		
cv2.imwrite("test3_sobel_scaled.jpg", sobelResults)
cv2.imwrite("test3_prewitt_scaled.jpg", prewittResults)

#thinning of Sobel and Prewitt edges
maxHorizontalEdgeStrSobel = maxVerticalEdgeStrSobel = 0
maxHorizontalEdgeStrPrewitt = maxVerticalEdgeStrPrewitt = 0
r , c = sobelResults.shape
sobelResultsThinned = prewittResultsThinned = np.zeros([r,c])

# Thinning
rows, columns = sobelResults.shape
for r in range(1, rows-1):
	for c in range(1, columns-1):
		# local maxima
		edgeSobel = sobelResults[r][c]
		edgePrewitt = prewittResults[r][c]
		if ((edgeSobel >= sobelResults[r][c-1] and edgeSobel >= sobelResults[r][c+1]) or (edgeSobel >= sobelResults[r-1][c] and edgeSobel >= sobelResults[r+1][c])):
			sobelResultsThinned[r][c] = edgeSobel
		if ((edgePrewitt >= prewittResults[r][c-1] and edgePrewitt >= prewittResults[r][c+1]) or (edgePrewitt >= prewittResults[r-1][c] and edgePrewitt >= prewittResults[r+1][c])):	
			prewittResultsThinned[r][c] = edgePrewitt		
cv2.imwrite("test3_sobel_scaled_thinned.jpg", sobelResultsThinned)
cv2.imwrite("test3_prewitt_scaled_thinned.jpg", prewittResultsThinned)			
				 

