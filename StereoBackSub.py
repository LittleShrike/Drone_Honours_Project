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
    print "Gathering background data..."
    backSubGMG = cv2.bgsegm.createBackgroundSubtractorGMG(20, .8) 
    backSubGMG.setDefaultLearningRate(.005)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    print "Background Subtraction GMG settings"
    print "Decision Threshold: {}" .format(backSubGMG.getDecisionThreshold())
    print "Learning Rate: {}" .format(backSubGMG.getDefaultLearningRate())
    return backSubGMG, kernel

###############
# GMM - expectation maximization algorithm
###############
def defGMMBackSub(background):
    em = cv2.ml.EM_create()
    em.train(background)
    em.setClustersNumber(5)
    
    return em
"""    
###############
#Distance Calculator
###############
def CamDistance(knownWidth, focalLength, perWidth):
    #Compute distance
    return (knownWidth * focalLength) / perWidth

"""
##############
#Find Drone
##############
def stereoFindDrone(imageL, imageR, fgbg, kernel, camMatrix, distCoeffs):    
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))    
    #cv2.imshow("L", imageL)
    #cv2.imshow("R", imageR)
    #cv2.waitKey(0)  
    unDistImgL = cv2.undistort(imageL, camMatrix[0], distCoeffs[0])
    unDistImgR = cv2.undistort(imageR, camMatrix[1], distCoeffs[1])    
    
    combinedImage = [unDistImgL , unDistImgR]   #Combine Images into array
    
    bothFound = False #Find contour area in both cameras
    leftCam = True #False = Left | True = Right
        
    droneContL, droneContR, noNoiseL, noNoiseR = None, None, None, None     
     
    for image in combinedImage: #Loop through images
        fgmask = fgbg.apply(image) #Apply backgroundsubtraction mask
        noNoise = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  #Process noise   
        blowNoise = cv2.dilate(noNoise, None, iterations=2)     #Process noise 
        _, contNoNoise, _ = cv2.findContours(blowNoise.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     
        sortedCont = sorted(contNoNoise, key = cv2.contourArea, reverse = True)[:5] #[:10]    POSSIBLE OPTIMISATION IMPROVEMENTS CAN BE MADE HERE    
        tempCont = []

        for c in sortedCont:    #Loop through contours in image           
            #Approximate contour
            peri = cv2.arcLength(c, True)            
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)          
            area = cv2.contourArea(c)
    
            if (area < 250): 
                #print("continue")                
                continue
            elif (len(approx) > 7 and len(approx) < 20):
                #print("FOUND ONE!!!!!!!!!!!!!!!!!!!!!!!!")                
                tempCont = approx                
                if leftCam == False:
                    bothFound = True
                break
        if leftCam == True and len(tempCont) > 0: #Left
            #if len(tempCont)            
            droneContL = tempCont[:7]
            noNoiseL = noNoise
            leftCam = False
        elif leftCam == False and len(tempCont) > 0: #right
            droneContR = tempCont[:7]       
            noNoiseR = noNoise
        #else:
        #    noNoiseL, noNoiseR  = noNoise, noNoise
    
    #return bothFound, droneContL, droneContR        
    return bothFound, droneContL, noNoiseL, droneContR, noNoiseR #, cv2.fitEllipse([droneCont])
    """
    cv2.drawContours(imageL, [droneContL], -1, (0, 255, 255), 3)    
    cv2.drawContours(imageR, [droneContR], -1, (0, 255, 255), 3)  
    
    return imageL, imageR
    #return cv2.minAreaRect()

"""
###########
# Main
###########
"""
def stereoBackSub(vidStreamL, vidStreamR):
print ("Background subtraction start!")

#cap = cv2.VideoCapture(0)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()    

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
fgbg3 = defGMGBackSub()
#maiorArea = 0

######
#firstFrame = None

while(1):
    #ret, frame = cap.read()

    retL, frameL = vidStreamL.read()
    retR, frameR = vidStreamR.read()
    
    #frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGrayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    frameGrayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    #droneCont, noNoise = FindDrone(frameGray, fgbg3)
    droneCont, noNoise = FindDrone(frameGrayL, fgbg3)
    droneCont, noNoise = FindDrone(frameGrayR, fgbg3)
        
    #focalLength = (droneCont[1][0] * ) 
    colourFrame = cv2.cvtColor(noNoise, cv2.COLOR_GRAY2BGR)
    
    cv2.drawContours(colourFrame, [droneCont], -1, (0, 255, 255), 3)    
    cv2.drawContours(frame, [droneCont], -1, (0, 255, 0), 3)    
    
    #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)        
    #cv2.drawContours(frame, [droneCont], -1, (0, 255, 0), 3)
    cv2.imshow("Drone", colourFrame)
    cv2.imshow("frame", frame)for c in Temp:
        
        #Approximate contour
        print(c)
        periL = cv2.arcLength(c[0], True) #Left
        periR = cv2.arcLength(c[1], True) #Right       
        
        approxL = cv2.approxPolyDP(c[0], 0.02 * periL, True)
        approxR = cv2.approxPolyDP(c[1], 0.02 * periR, True)     
        
        areaL = cv2.contourArea(c[0])
        areaR = cv2.contourArea(c[1])

        #print("Area {}").format(area)
        if (areaL < 200 or areaR < 200): #POSSIBLY DANGEROUS
            continue
        elif (len(approxL) > 7 and len(approxL) < 20) and (len(approxR) > 7 and len(approxR) < 20):
            droneContL = approxL
            droneContR = approxR
            #cnt = approx[:4]
            break            
    #cv2.imshow('img Contours', img)    


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
"""