# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:47:53 2017

@author: william
"""
import sys
import cv2
import numpy as np

#import OpenCVHacks as hax

####################
# GMG - Background Subtraction setup
####################
def defGMGBackSub():
    backSubGMG = cv2.bgsegm.createBackgroundSubtractorGMG(100, .8) 
    backSubGMG.setDefaultLearningRate(.005)
    
    print("Background Subtraction GMG settings")
    print("Decision Threshold: {}").format(backSubGMG.getDecisionThreshold())
    print("Learning Rate: {}").format(backSubGMG.getDefaultLearningRate())
    return backSubGMG

###############
# GMM - expectation maximization algorithm
###############
def defGMMBackSub(background):
    em = cv2.ml.EM_create()
    em.train(background)
    em.setClustersNumber(5)
    
    return em
    
#def Samples(img):
    

###########
# Main
###########

print ("Program start!")

"""
# Read in two images
if len(sys.argv) < 2:
        print "No filenames specified"
        print "USAGE: find_obj.py <Background Image>"
        sys.exit(1)

BackN = sys.argv[1]
backgroundImg = cv2.imread(BackN, 0)
DroneN = sys.argv[2]
droneImg = cv2.imread(DroneN, 0)

if backgroundImg is None:
        print 'Failed to load background:', backgroundImg
        sys.exit(1)

if droneImg is None:
    print 'Failed to load droneimg:', droneImg
    sys.exit(1)

img = backgroundImg - droneImg
cv2.imshow('Subtrackted', img)
cv2.waitKey(0)

"""
cap = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()    
#fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG2() 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
fgbg3 = defGMGBackSub()
maiorArea = 0

######
firstFrame = None


"""####

cv2::Mat data = imread("Background2.jpg", False)
cv2::Mat orig_img, bin_img
pattern = cv2.CreateMat(0, 0, CV_32F)
em_GMM = defGMMBackSub(DownImg)

for r in range(0, data.rows):
    for c in range(0, data.cols):
        
######"""

while(1):
    ret, frame = cap.read()
    
    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    #orig_img = framegray.copy()
    #bin_imh = cv2::Mat(orig_img.rows, orig_img.cols, CV_8U, cv2)
  #####
    #hax.DebugPointer(frame)
    #cv2.imshow('dbugframe', frame)
    #cv2.waitKey(0)
    #fgmask = fgbg.apply(framegray)
    #fgmask2 = fgbg2.apply(frame)
    #noNoise = cv2.bilateralFilter(fgmask3, 11, 17, 17)    
    #memSto = cv2.CreateMemStorage(0)
    #contNoNoise = cv2.findContours(noNoise, cv2.CV_RETR_LIST, cv2.CV_LINK_RUNS)

    fgmask3 = fgbg3.apply(framegray)
    noNoise = cv2.morphologyEx(fgmask3, cv2.MORPH_OPEN, kernel)
    blowNoise = cv2.dilate(noNoise, None, iterations=2)    
    img, contNoNoise, hierarchy = cv2.findContours(blowNoise.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    sortedCont = sorted(contNoNoise, key = cv2.contourArea, reverse = True)[:5] #[:10]    
    cnt, droneCont = None, None    
# loop through conts    
    for c in sortedCont:
        #Approximate contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        #print("Area {}").format(area)
        if (area < 200):
            continue
        elif len(approx) > 7 and len(approx) < 20:
            droneCont = approx
            cnt = approx[:4]
            break
    colourFrame = cv2.cvtColor(noNoise, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(colourFrame, [droneCont], -1, (0, 255, 255), 3)
    
    
    cv2.drawContours(frame, [droneCont], -1, (0, 255, 0), 3)    
    
    #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)        
    #cv2.drawContours(frame, [droneCont], -1, (0, 255, 0), 3)
    cv2.imshow("Drone", colourFrame)
    cv2.imshow("frame", frame)
    #cv2.imshow('img Contours', img)    
    """
    for cnt in contNoNoise:
        box = cv2.boundingRect(cnt)
        if(cv2.contourArea(cnt) > maiorArea):
            maiorArea = cv2.contourArea(cnt)
            rectanglePoints = box
    
    p1 = (rectanglePoints[0], rectanglePoints[1])
    p2 = (rectanglePoints[0] + rectanglePoints[2], rectanglePoints[1] + rectanglePoints[3])

    cv2.Rectangle(frame, p1, p2, cv2.CV_RGB(0,0,0), 2)
    cv2.Rectangle(noNoise, p1, p2, cv.cv_RGB(225,225,225), 1)

    temp1 = p2[0] - p1[0]
    temp2 = p2[1] - p1[1]
    
    cv2.Line(noNoise, (p1[0]+temp1/2, p1[1]), (p1[0]+temp1/2,p2[1]), cv2.CV_RGB(255,255,255), 1)
    cv2.Line(noNoise, (p1[0], p1[1]+temp2/2), (p2[0], p1[1]+temp2/2), cv2.CV_RGB(255,255,255), 1)  
    """
    #cv2.imshow('frame', frame)
    #cv2.imshow('fgmask', fgmask)
    #cv2.imshow('MOG2', fgmask2)
    #cv2.imshow('GMG', fgmask3)
    #cv2.imshow('noNoise', noNoise)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()

cv2.destroyAllWindows()
