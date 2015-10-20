import os
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la
import cv2

def quatmult(q1,q2):
    # quaternion multiplication
    result = np.zeros(4)
    result[0] = (q1[0] * q2[0]) - (q1[1] * q2[1]) - (q1[2] * q2[2]) - (q1[3] * q2[3])
    result[1] = (q1[0] * q2[1]) + (q1[1] * q2[0]) + (q1[2] * q2[3]) - (q1[3] * q2[2])
    result[2] = (q1[0] * q2[2]) - (q1[1] * q2[3]) + (q1[2] * q2[0]) + (q1[3] * q2[1])
    result[3] = (q1[0] * q2[3]) + (q1[1] * q2[2]) - (q1[2] * q2[1]) + (q1[3] * q2[0])
    return result
    
def getQuaternionConjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])    

# given angle of rotation in degrees and axis of rotation
# return a quaternion to represent the rotation
def rotationQuaternion(rotationAngle,axisOfRotation):
    rotationAngleRadian = np.radians(rotationAngle)
    result = np.zeros(4)
    result[0] = np.cos(rotationAngleRadian/2)
    result[1] = np.sin(rotationAngleRadian/2) * axisOfRotation[0]
    result[2] = np.sin(rotationAngleRadian/2) * axisOfRotation[1]
    result[3] = np.sin(rotationAngleRadian/2) * axisOfRotation[2]
    return result

# given an input quaternion
# return corresponding 3x3 rotation matrix
def quat2rot(q):
    rotationMatrix = np.zeros([3,3])
    rotationMatrix[0][0] = (np.power(q[0],2) + np.power(q[1],2) - np.power(q[2],2) - np.power(q[3],2))
    rotationMatrix[0][1] = 2 * ((q[1] * q[2]) - (q[0] * q[3]))
    rotationMatrix[0][2] = 2 * ((q[1] * q[3]) + (q[0] * q[2]))
    rotationMatrix[1][0] = 2 * ((q[1] * q[2]) + (q[0] * q[3]))
    rotationMatrix[1][1] = (np.power(q[0],2) + np.power(q[2],2) - np.power(q[1],2) - np.power(q[3],2))
    rotationMatrix[1][2] = 2 * ((q[2] * q[3]) - (q[0] * q[1]))
    rotationMatrix[2][0] = 2 * ((q[1] * q[3]) - (q[0] * q[2]))
    rotationMatrix[2][1] = 2 * ((q[2] * q[3]) + (q[0] * q[1]))
    rotationMatrix[2][2] = (np.power(q[0],2) + np.power(q[3],2) - np.power(q[1],2) - np.power(q[2],2))
    return np.matrix(rotationMatrix)

# returning a camera translation from its quaternion    
def getCameraTranslation(translation):
    return translation[1:4]    
    
def perspectiveProjection(scenePts,translation,orientation):
    u0, v0, bu, bv, ku, kv, focal_length = 0, 0, 1, 1, 1, 1, 1
    perspectivePts = []
    currentTranslation = getCameraTranslation(translation)
    for pts in scenePts:
        difference = pts - currentTranslation
        u = ((focal_length*(np.dot(difference,orientation[0].T)) * bu) / (np.dot(difference,orientation[2].T))) + u0
        v = ((focal_length*(np.dot(difference,orientation[1].T)) * bv) / (np.dot(difference,orientation[2].T))) + v0
        perspectivePts.append((u,v))
    return perspectivePts

def orthographicProjection(scenePts,translation,orientation):
    u0, v0, bu, bv, = 0, 0, 1, 1
    orthographicPts = []
    currentTranslation = getCameraTranslation(translation)
    for pts in scenePts:
        difference = pts - currentTranslation
        u = (np.dot(difference,orientation[0].T)* bu) + u0
        v = (np.dot(difference,orientation[1].T)* bv) + u0
        orthographicPts.append((u,v))
    return orthographicPts

def plot(points,frame,figure):
    plt.figure(figure)
    plt.subplot(2,2,frame)
    xPoints = []
    yPoints = []
    for i, (x, y) in enumerate(points):
        xPoints.append(x.item(0))
        yPoints.append(y.item(0))
    plt.plot(xPoints, yPoints)
    plt.title('Frame {}'.format(frame))
    
def determiningHomographyMatrix(points,projectionPts,pointsToUse):
    findHomography = []
    size = pointsToUse.size
    for i in range(0,5):
        index = pointsToUse[i]
        uc = projectionPts[index][0].item(0)
        vc = projectionPts[index][1].item(0)
        up = points[index][0]
        vp = points[index][1]
        row1 = np.array([up,vp,1,0,0,0,-(uc)*up,-(uc)*vp,-uc])
        row2 = np.array([0,0,0,up,vp,1,-(vc)*up,-(vc)*vp,-vc])
        findHomography.append(row1)
        findHomography.append(row2)
    return findHomography
                        
# Defining the shape
pts = np.zeros([11,3])
pts[0,:] = [-1,-1,-1]
pts[1,:] = [1,-1,-1]
pts[2,:] = [1,1,-1]
pts[3,:] = [-1,1,-1]
pts[4,:] = [-1,-1,1]
pts[5,:] = [1,-1,1]
pts[6,:] = [1,1,1]
pts[7,:] = [-1,1,1]
pts[8,:] = [-0.5,-0.5,-1]
pts[9,:] = [0.5,-0.5,-1]
pts[10,:] = [0,0.5,-1]

# camera starting position
cameraTranslation1 = np.array([0,0,-5])
frame1Quaternion = np.array([0,0,0,-5])
yAxis = np.array([0,1,0])
rotateQuaternion = rotationQuaternion(-30, yAxis)
rotateQuaternionConjugate = getQuaternionConjugate(rotateQuaternion)
# For camera translation
# Using quaternion multiplication
# p' = qpq*
frame2Quaternion = quatmult(quatmult(rotateQuaternion,frame1Quaternion),rotateQuaternionConjugate)
frame3Quaternion = quatmult(quatmult(rotateQuaternion,frame2Quaternion),rotateQuaternionConjugate)
frame4Quaternion = quatmult(quatmult(rotateQuaternion,frame3Quaternion),rotateQuaternionConjugate)
# For camera orientation
# Using rotation Matrix
# r' = R(q)r
quatmat_1 = np.matrix(np.identity(3))
rotateQuaternionOrientation = rotationQuaternion(30, yAxis)
rotateMatrix = quat2rot(rotateQuaternionOrientation)
quatmat_2 = quatmat_1 * rotateMatrix
quatmat_3 = quatmat_2 * rotateMatrix
quatmat_4 = quatmat_3 * rotateMatrix

perspectiveProjectionFrame1 = perspectiveProjection(pts,frame1Quaternion,quatmat_1)
perspectiveProjectionFrame2 = perspectiveProjection(pts,frame2Quaternion,quatmat_2)
perspectiveProjectionFrame3 = perspectiveProjection(pts,frame3Quaternion,quatmat_3)
perspectiveProjectionFrame4 = perspectiveProjection(pts,frame4Quaternion,quatmat_4)
plot(perspectiveProjectionFrame1,1,1)
plot(perspectiveProjectionFrame2,2,1)
plot(perspectiveProjectionFrame3,3,1)
plot(perspectiveProjectionFrame4,4,1)
plt.show(1)

orthographicProjectionFrame1 = orthographicProjection(pts,frame1Quaternion,quatmat_1)
orthographicProjectionFrame2 = orthographicProjection(pts,frame2Quaternion,quatmat_2)
orthographicProjectionFrame3 = orthographicProjection(pts,frame3Quaternion,quatmat_3)
orthographicProjectionFrame4 = orthographicProjection(pts,frame4Quaternion,quatmat_4)
plot(orthographicProjectionFrame1,1,2)
plot(orthographicProjectionFrame2,2,2)
plot(orthographicProjectionFrame3,3,2)
plot(orthographicProjectionFrame4,4,2)
plt.show(2)

pointsToUse = np.array([0,1,2,3,8])
findHomography = determiningHomographyMatrix(pts,perspectiveProjectionFrame3,pointsToUse)
u,s,v = la.svd(findHomography)
rows, columns = v.shape
homography = v[rows-1]/v[rows-1][columns-1]
print homography

    
    















    