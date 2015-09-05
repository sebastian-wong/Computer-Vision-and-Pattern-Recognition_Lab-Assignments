import numpy as np
import cv2
from matplotlib import pyplot as plt

## File path
## os.chdir("C:/Users/Sebastian/Documents/GitHub/Computer-Vision-and-Pattern-Recognition_Lab-Assignments/A0101856J_SebastianWong_Lab2/pic/pic")

# read an image in greyscale
img = cv2.imread("pic1.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
# Computes values of histogram and occurrences of input data
# that fall within each bin
hist,bins = np.histogram(img.flatten(),256,[0,256])
# Computing cummulative distribution function
cdf = hist.cumsum()
# Normalizing cummulative distribution function
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'g')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img1 = cdf[img]
# Computes new array values of histogram
hist,bins = np.histogram(img1.flatten(),256,[0,256])
cdfEqualized = hist.cumsum()
cdfEqualizedNormalised = cdfEqualized * hist.max()/cdfEqualized.max()
plt.plot(cdfEqualizedNormalised, color = 'g')
plt.hist(img1.flatten(),256,[0,256],color = 'r')
plt.xlim([0,256])
plt.legend(('cdf of equalized image', 'histogram after equalization'), loc = 'upper left')
plt.show()
cv2.imwrite("pic1_equalized.jpg",img1)