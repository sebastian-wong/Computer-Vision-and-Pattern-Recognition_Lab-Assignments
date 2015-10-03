import os
import cv2
import cv2.cv as cv
import numpy as np


def convertToIntegers(frameWidth, frameHeight, framesPerSecond, frameCount):
    frameWidth = int(frameWidth)
    frameHeight = int(frameHeight)
    framesPerSecond = int(framesPerSecond)
    frameCount = int(frameCount)
    return frameWidth, frameHeight, framesPerSecond, frameCount

def backgroundExtraction(cap,frameCount):
    ret,image = cap.read()
    avgImage = np.float32(image)
    normImg = 0
    for frame in range (1,frameCount):
        ret, image = cap.read()
        avgImage = (frame/(frame+1.0))*avgImage + (1.0/(frame+1.0))*image
        normImg = cv2.convertScaleAbs(avgImage)
        cv2.imshow('image',image)
        cv2.imshow('normImg', normImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    cap.release()    
    return normImg    
           
# Reading the video
cap = cv2.VideoCapture(os.getcwd() + "/videos/traffic.mp4")
frameWidth = cap.get(3)
frameHeight = cap.get(4)
framesPerSecond = cap.get(5)
frameCount = cap.get(7)
frameWidthInt, frameHeightInt, framesPerSecondInt, frameCountInt = convertToIntegers(frameWidth, frameHeight, framesPerSecond, frameCount)
background = backgroundExtraction(cap,frameCountInt)
cv2.imwrite(os.getcwd() + "/results/background.jpg", background)

