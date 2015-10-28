import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def gauss_kernels(size,sigma=1):
    ## returns a 2d gaussian kernel
    if size<3:
        size = 3
    m = size/2
    x, y = np.mgrid[-m:m+1, -m:m+1]
    kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    kernel_sum = kernel.sum()
    if not sum==0:
        kernel = kernel/kernel_sum
    return kernel    
            
def getEdgeStrength(image):
    # Setting Sobel kernel
    # Horizontal kernel detects vertical edges
    # Vertical kernel detects horizontal edges
    sobelHorizontal = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobelVertical = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    imgHeight, imgWidth = image.shape
    gx = np.zeros([imgHeight,imgWidth])
    gy = np.zeros([imgHeight,imgWidth])
    for x in range(0, imgHeight-2):
    	for y in range(0, imgWidth-2):
            horizontalEdgeStrength = 0
            verticalEdgeStrength = 0
            
            verticalEdgeStrength = ((sobelHorizontal[0][0] * img[x][y]) + (sobelHorizontal[0][1] * img[x][y+1]) + (sobelHorizontal[0][2] * img[x][y+2]) + 
		    (sobelHorizontal[1][0] * img [x+1][y]) + (sobelHorizontal[1][1] * img[x+1][y+1]) + (sobelHorizontal[1][2] * img[x+1][y+2]) + 
		    (sobelHorizontal[2][0] * img[x+2][y]) + (sobelHorizontal[2][1] * img[x+2][y+1]) + (sobelHorizontal[2][2] * img[x+2][y+2]))
        
            horizontalEdgeStrength = ((sobelVertical[0][0] * img[x][y]) + (sobelVertical[0][1] * img[x][y+1]) + (sobelVertical[0][2] * img[x][y+2]) + 
		    (sobelVertical[1][0] * img [x+1][y]) + (sobelVertical[1][1] * img[x+1][y+1]) + (sobelVertical[1][2] * img[x+1][y+2]) + 
		    (sobelVertical[2][0] * img[x+2][y]) + (sobelVertical[2][1] * img[x+2][y+1]) + (sobelVertical[2][2] * img[x+2][y+2]))
            
            gx[x+1][y+1] = horizontalEdgeStrength
            gy[x+1][y+1] = verticalEdgeStrength
    return gx, gy
    
def productOfDerivatives(horizontalEdgeStrengths,verticalEdgeStrengths):
    I_xx = horizontalEdgeStrengths * horizontalEdgeStrengths
    I_xy = horizontalEdgeStrengths * verticalEdgeStrengths
    I_yy = verticalEdgeStrengths * verticalEdgeStrengths
    return I_xx, I_xy, I_yy

def convolution(derivativeXX, derivativeXY, derivativeYY, kernel):
    height,width = derivativeXX.shape
    W_xx = np.zeros([height,width])
    W_xy = np.zeros([height,width])
    W_yy = np.zeros([height,width])
    for row in range(0, height-2):
        for column in range(0, width-2):
            xx = xy = yy = 0
            
            xx = ((kernel[0][0] * derivativeXX[row][column]) + (kernel[0][1] * derivativeXX[row][column+1]) + (kernel[0][2] * derivativeXX[row][column+2]) +
            (kernel[1][0] * derivativeXX[row+1][column]) + (kernel[1][1] * derivativeXX[row+1][column+1]) + (kernel[1][2] * derivativeXX[row+1][column+2]) +
            (kernel[2][0] * derivativeXX[row+2][column]) + (kernel[2][1] * derivativeXX[row+2][column+1]) + (kernel[2][2] * derivativeXX[row+2][column+2])
            )
            
            xy = ((kernel[0][0] * derivativeXY[row][column]) + (kernel[0][1] * derivativeXY[row][column+1]) + (kernel[0][2] * derivativeXY[row][column+2]) +
            (kernel[1][0] * derivativeXY[row+1][column]) + (kernel[1][1] * derivativeXY[row+1][column+1]) + (kernel[1][2] * derivativeXY[row+1][column+2]) +
            (kernel[2][0] * derivativeXY[row+2][column]) + (kernel[2][1] * derivativeXY[row+2][column+1]) + (kernel[2][2] * derivativeXY[row+2][column+2])
            )
            
            yy = ((kernel[0][0] * derivativeYY[row][column]) + (kernel[0][1] * derivativeYY[row][column+1]) + (kernel[0][2] * derivativeYY[row][column+2]) +
            (kernel[1][0] * derivativeYY[row+1][column]) + (kernel[1][1] * derivativeYY[row+1][column+1]) + (kernel[1][2] * derivativeYY[row+1][column+2]) +
            (kernel[2][0] * derivativeYY[row+2][column]) + (kernel[2][1] * derivativeYY[row+2][column+1]) + (kernel[2][2] * derivativeYY[row+2][column+2])
            )
            
            W_xx[row+1][column+1] = xx
            W_xy[row+1][column+1] = xy
            W_yy[row+1][column+1] = yy
    return W_xx, W_xy, W_yy        

def computeHarrisCornerResponse(w_xx,w_xy,w_yy):
    height,width = w_xx.shape
    response = np.zeros([height,width])
    arrayW = np.zeros([2,2])
    maxResponse = 0
    for row in range(0,height):
        for column in range(0,width):
            arrayW[0][0] = w_xx[row][column]
            arrayW[0][1] = w_xy[row][column]
            arrayW[1][0] = w_xy[row][column]
            arrayW[1][1] = w_yy[row][column]
            determinantW = computeDeterminant(arrayW)
            traceW = computeTrace(arrayW)
            response[row][column] = (determinantW - (0.06*(traceW * traceW)))
            if (response[row][column] > maxResponse):
                maxResponse = response[row][column]
    return response , maxResponse        
            
def computeDeterminant(array):
    return ((array[0][0] * array[1][1]) - (array[0][1] * array[1][0])) 
            
def computeTrace(array):
    return (array[0][0] + array[1][1])

def filterResponse(responseArray,threshold):
    height,width = responseArray.shape
    for row in range(0,height):
        for column in range(0,width):
            if (responseArray[row][column] < 0.1*(threshold)):
                responseArray[row][column] = 0
    return responseArray            

def plotHarrisCornerResponse(image, responses):
    height, width, channels = image.shape
    magentaMarks = np.array([255,0,255])
    for row in range(0,height):
        for column in range(0,width):
            if (responses[row][column] != 0):
                 image[row][column] = magentaMarks       
    plt.figure()
    plt.imshow(image)        
    plt.show()
    return image            

def bgrToRgb(image):
    height, width, channels = image.shape
    newImage = np.zeros([height,width,channels])
    newImage[:,:,0] = image[:,:,2]
    newImage[:,:,1] = image[:,:,1]     
    newImage[:,:,2] = image[:,:,0]
    newImage = newImage.astype(np.uint8)                    
    return newImage
    
def rgbToBgr(image):
    height, width, channels = image.shape
    newImage = np.zeros([height,width,channels])
    newImage[:,:,0] = image[:,:,2]
    newImage[:,:,1] = image[:,:,1]     
    newImage[:,:,2] = image[:,:,0]
    newImage = newImage.astype(np.uint8)                    
    return newImage         
                                            
# Reading in original image
imgColoured = cv2.imread(os.getcwd() + "/Pictures/stitched.jpg")                                            
# Reading image in grayscale for processing
img = cv2.imread(os.getcwd() + "/Pictures/stitched.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
gx , gy = getEdgeStrength(img)
I_xx, I_xy, I_yy = productOfDerivatives(gx,gy)
gaussianKernel = gauss_kernels(3,1)
W_xx, W_xy, W_yy = convolution(I_xx,I_xy,I_yy,gaussianKernel)
harrisCornerReponse , maxResponse = computeHarrisCornerResponse(W_xx,W_xy,W_yy)
filteredHarrisCorner = filterResponse(harrisCornerReponse,maxResponse)
# Convert BGR to RGB for plt.show()
rgbImg = bgrToRgb(imgColoured)
result = plotHarrisCornerResponse(rgbImg,filteredHarrisCorner)
# Convert back to BGR for saving
result = rgbToBgr(result)
cv2.imwrite(os.getcwd() + "/Results/stitched1.jpg", result)














            