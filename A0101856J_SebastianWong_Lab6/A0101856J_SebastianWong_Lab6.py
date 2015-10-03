import os
import cv2
import cv2.cv as cv
import numpy as np


def convertToIntegers(frameWidth, frameHeight, framesPerSecond, frameCount):
    frameWidth = int(frameWidth)
    frameHeight = int(frameHeight)
    framesPerSecond = int(framesPerSecond)
    frameCount = int(framesCount)
    return frameWidth, frameHeight, framesPerSecond, frameCount

def backgroundExtraction(frameCount):
    ret,imageFrame = cap.read()
    avgImageFrame = np.float32(imageFrame)
    for imgFrame in range (1,frameCount):
        ret, imageFrame = cap.read()
        avgImageFrame = avgImageFrame + imageFrame
    avgImageFrame = avgImageFrame/frameCount
    return avgImageFrame    
           
# Reading the video
cap = cv2.VideoCapture(os.getcwd() + "/videos/traffic.mp4")
frameWidth = cap.get(3)
frameHeight = cap.get(4)
framesPerSecond = cap.get(5)
frameCount = cap.get(7)
frameWidthInt, frameHeightInt, framesPerSecondInt, frameCountInt = convertToIntegers(frameWidth, frameHeight, framesPerSecond, frameCount)
background = backgroundExtraction(frameCountInt)
cv2.imwrite(os.getcwd() + "/results/background.jpg", background)


