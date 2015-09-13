import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def rgbToHsv(image):
    rows , columns , channels = image.shape
    hsvImage = np.zeros([rows,columns,channels]) 
    for r in range(0,rows):
        for c in range(0, columns):
            # Initialise variables
            hue,saturation,value = 0,0,0
            blue , green , red = image[r,c]/255.0
            rgbMax = max(blue,green,red)
            rgbMin = min(blue,green,red)
            difference = rgbMax - rgbMin
            value = rgbMax
            # Computing saturation
            # Value is 0, blackz
            if (rgbMax == 0):
                saturation = 0
            else:
                saturation = difference/rgbMax		
            # Compute hue
            if (difference == 0):
                hue = 0
            elif (rgbMax == red):
                hue = 60 *(((green-blue)/difference)%6)
            elif (rgbMax == green):
                hue = 60 *(((blue-red)/difference) +2)
            elif (rgbMax == blue):
                hue = 60 * (((red-green)/difference) + 4)
            hsvImage[r][c] = [hue,saturation,value]     
    return hsvImage
    
def hsvToRgb(hsvImage):
    rows, columns, channels = hsvImage.shape
    rgbImage = np.zeros([rows,columns,channels])
    for r in range(0,rows):
        for c in range(0,columns):
            # Initialise variables
            blue, green, red, C, X, m = 0, 0, 0, 0, 0, 0
            hue, saturation, value = hsvImage[r,c]
            C = value * saturation
            X = C * (1 - abs(((hue/60.0)%2) - 1))
            m = value - C
            if (0 <= hue < 60):
                red = C
                green = X
            elif (60 <= hue < 120):
                red = X
                green = C
            elif (120 <= hue < 180):
                green = C
                blue = X
            elif (180 <= hue < 240):
                green = X
                blue = C
            elif (240 <= hue <= 300):
                red = X
                blue = C        
            elif (300 <= hue < 360):
                red = C
                blue = X    
            rgbImage[r,c] = [(blue + m) * 255, (green + m) * 255, (red + m) * 255]
    return rgbImage
    
def histogramEqualization(image):
    image = image.astype(int)
    hist,bins = np.histogram(image.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdfMasked = np.ma.masked_equal(cdf,0)
    cdfMasked = (cdfMasked - cdfMasked.min())*255/(cdfMasked.max()-cdfMasked.min())
    cdf = np.ma.filled(cdfMasked,0).astype('uint8')
    imgEqualized = cdf[image]
    return imgEqualized
    
    
# Reading flower image                                       
imgFlower = cv2.imread(os.getcwd() + "/Pictures/flower.jpg")
# Converting flower from rgb colour space to hsv  
hsvFlower = rgbToHsv(imgFlower)

# imshow expects [0,1] for floating point and [0,255] for unsigned chars
# imwrite always expects [0,255]

# divide by 360 to normalise to [0,1]
cv2.imshow('hue component', hsvFlower[:,:,0]/360.0)
cv2.waitKey(0)
cv2.imwrite( os.getcwd()+ '/Results/hue.jpg', hsvFlower[:,:,0])

cv2.imshow('saturation component', hsvFlower[:,:,1])
cv2.waitKey(0)
cv2.imwrite( os.getcwd()+ '/Results/saturation.jpg', hsvFlower[:,:,1]*255.0)

cv2.imshow('brightness component', hsvFlower[:,:,2])
cv2.waitKey(0)
cv2.imwrite( os.getcwd()+ '/Results/brightness.jpg', hsvFlower[:,:,2]*255.0)

cv2.destroyAllWindows()
# Converting flower from hsv colour space to rgb
rgbFlower = hsvToRgb(hsvFlower)
cv2.imwrite( os.getcwd()+ '/Results/hsv2rgb.jpg', rgbFlower)
# Reading in bee image
imgBee = cv2.imread(os.getcwd() + "/Pictures/bee1.png")
# Converting bee from rgb colour space to hsv
hsvBee = rgbToHsv(imgBee)
valueComponent = hsvBee[:,:,2]*255.0
hsvBee[:,:,2] = (histogramEqualization(valueComponent))/255.0
rgbBee = hsvToRgb(hsvBee)
cv2.imwrite(os.getcwd() + "/Results/histeq.jpg", rgbBee)

















