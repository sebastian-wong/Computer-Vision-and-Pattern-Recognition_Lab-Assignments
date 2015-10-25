
import os
import cv2
import math
import numpy as np
import utils

from numpy import linalg

def filter_matches(matches, ratio = 0.75):
    filtered_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])
    
    return filtered_matches

def imageDistance(matches):

    sumDistance = 0.0

    for match in matches:

        sumDistance += match.distance

    return sumDistance

def findDimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0,0]
    base_p2[:2] = [x,0]
    base_p3[:2] = [0,y]
    base_p4[:2] = [x,y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

        if ( max_x == None or normal_pt[0,0] > max_x ):
            max_x = normal_pt[0,0]

        if ( max_y == None or normal_pt[1,0] > max_y ):
            max_y = normal_pt[1,0]

        if ( min_x == None or normal_pt[0,0] < min_x ):
            min_x = normal_pt[0,0]

        if ( min_y == None or normal_pt[1,0] < min_y ):
            min_y = normal_pt[1,0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)
    

def imageStitching(baseImageBGR,nextImageBGR):
    
    baseImage = cv2.cvtColor(baseImageBGR, cv2.COLOR_BGR2GRAY)
    nextImage = cv2.cvtColor(nextImageBGR, cv2.COLOR_BGR2GRAY)
    # Using SURF feature detector
    detector = cv2.SURF(8000)
    # Finding keypoints
    baseFeatures, baseDescs = detector.detectAndCompute(baseImage,None)
    
    # Parameters for nearest-neighbor matching
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, 
        trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    
    imageToStitch = cv2.GaussianBlur(nextImage)
    
    
    
    

           
# Reading in videos
capLeft = cv2.VideoCapture(os.getcwd() + "/their_football_videos/left_camera.mov")
capCentre = cv2.VideoCapture(os.getcwd() + "/their_football_videos/centre_camera.mov")
capRight = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mov")
frameCounts = int(capLeft.get(7))
retLeft, imageLeft = videoLeft.read()
retCentre, imageCentre = videoCentre.read()
retRight, imageRight = videoRight.read()           


    
    
    
    