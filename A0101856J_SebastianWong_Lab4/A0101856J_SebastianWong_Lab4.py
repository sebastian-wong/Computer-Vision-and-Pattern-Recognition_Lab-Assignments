import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

imgFlower = cv2.imread(os.getcwd() + "Pictures/flower.jpg")
rows , columns , channels = imgFlower.shape
HSV = np.zeros([rows*columns,3])
for r in range(0,rows):
	for c in range(0, columns):
		b , g , r = imgFlower[r,c]/255.0
		rgbMax = max(b,g,r)
		rgbMin = min(b,g,r)
		difference = rgbMax - rgbMin
		value = rgbMax
		# Value is 0, black
		if (rgbMax == 0):
			hue = saturation = 0
			return (hue,saturation,value)
		# Compute saturation
		saturation = difference/rgbMax	
			
		# Compute hue
		if (rgbMax == r):
			hue = 60 *(((g-b)/difference)%6)
		elif (rgbMax == g):
			hue = 60 *(((b-r)/difference) +2)
		elif (rgbMax == b):
			hue = 60 * (((r-g)/difference) + 4)


