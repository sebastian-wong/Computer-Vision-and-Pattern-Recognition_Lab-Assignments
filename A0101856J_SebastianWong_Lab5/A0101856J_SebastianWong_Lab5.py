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
        kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma)) kernel_sum = kernel.sum()
        if not sum==0:
            kernel = kernel/kernel_sum return kernel

# Reading image
img = cv2.imread(os.getcwd() + "/Pictures/building1.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
            
# Setting Sobel kernel
sobelHorizontal = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobelVertical = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

            