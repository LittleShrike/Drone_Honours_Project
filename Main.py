import sys
import cv2
import imutils
import numpy as np
import StereoCalibrate
import StereoBackSub

vidStreamL = cv2.VideoCapture(0)
vidStreamR = cv2.VideoCapture(1)

"""
vidStreamL = cv2.VideoCapture("outputL.avi")
vidStreamR = cv2.VideoCapture("outputR.avi")
"""
GMGbs, kernel = StereoBackSub.defGMGBackSub()
#print(GMGbs)
#print(kernel)
#sys.settrace

CalibFile = "MainCalibration.txt"

#Which cameras connected calabrates the data
#StereoCalibrate.stereoCalabrate(vidStreamL, vidStreamR, CalibFile)

size, camMatrixL, distCoeffL, camMatrixR, distCoeffR, R, T, E, F = StereoCalibrate.importFile(CalibFile)
camMatrix = [camMatrixL, camMatrixR]
distCoeff = [distCoeffL, distCoeffR]
#print(camMatrix)
#print(distCoeff)
        
#R1/2 Rectification transform   | P1/2 Projection Matrix
#height, width = size[0], size[1]
#ret, frameL = vidStreamL.read()
#R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camMatrixL, distCoeffL, camMatrixR, distCoeffR, size, R, T, alpha=-1)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camMatrixL, distCoeffL, camMatrixR, distCoeffR, size, R, T, alpha=-1)

print("Projection Matrix")
print(P1)
print(P2)

while(1):
  
    retL, frameL = vidStreamL.read()
    retR, frameR = vidStreamR.read()
    
    if (retL == False or retR == False):
        print("Frames empty")
        break
    
    frameGrayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    frameGrayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    
    bothFound, droneContL, noNoiseL, droneContR, noNoiseR = StereoBackSub.stereoFindDrone(frameGrayL, frameGrayR, GMGbs, kernel, camMatrix, distCoeff)
    
    #droneL = np.array(droneContL)
    #droneR = np.array(droneContR)
    #imageL, imageR = StereoBackSub.stereoFindDrone(frameGrayL, frameGrayR, GMGbs, kernel)    
    #bothFound, droneContL, droneContR = StereoBackSub.stereoFindDrone(frameGrayL, frameGrayR, GMGbs, kernel)
    
    #droneContL = [[[639, 86]], [[  0, 132]], [[  0, 419]], [[ 91, 437]], [[  0, 479]], [[399 479]], [[339 313]]]
    #droneContR = [[[  8, 277]], [[  8, 289]], [[ 14, 288]], [[ 15, 282]], [[ 22, 280]], [[ 26 287]], [[ 37 288]]]
    #bothFound = True #Both droneContL and droneContR found

     
    if bothFound == True:
        """ Calculate disparity map"""
        #disparity = stereo.compute(frameGrayL, frameGrayR)        
        #cv2.imshow("Disparity", disparity)
        #print("DroneContours")
        #print(droneContL)    
        #print(droneContR) 
        

        #print(contL)
        
        
        M = cv2.moments(droneContL)
        cLX = int(M["m10"] / M["m00"])
        cLY = int(M["m01"] / M["m00"])
        
        M = cv2.moments(droneContR)
        cRX = int(M["m10"] / M["m00"])
        cRY = int(M["m01"] / M["m00"])
        contL = np.array((cLX, cLY), dtype=np.float).reshape(-1, 2)
        contR = np.array((cRX, cRY), dtype=np.float).reshape(-1, 2)
        #print(contL[0])        
        
        dronePos = cv2.triangulatePoints(P1, P2, contL[0], contR[0])
        
        #Triangulate Points will throw segmentation error
        #dronePos = cv2.triangulatePoints(P1, P2, droneContL, droneContR)
        
        print("Drone Position in 3D space")        
        print(dronePos)
        
        cv2.drawContours(frameL, [droneContL], -1, (0, 255, 255), 3)    
        cv2.drawContours(frameR, [droneContR], -1, (0, 255, 255), 3)   
        cv2.circle(frameL, (cLX, cLY), 1, (0, 0, 0), 3)    
        cv2.circle(frameL, (cRX, cRY), 1, (0, 0, 0), 3)  
        
        
     
    #cv2.imshow("noNoiseL", noNoiseL)
    #cv2.imshow("noNoiseR", noNoiseR)
    cv2.imshow("imageL", frameL)
    cv2.imshow("imageR", frameR)  

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

vidStreamL.release()    
vidStreamR.release()
cv2.destroyAllWindows()
