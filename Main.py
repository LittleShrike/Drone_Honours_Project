import sys
import cv2
import imutils
import math
import numpy as np
import time
import StereoCalibrate

#Detection methods implemented
import QRCode
import StereoBackSub
import Glyph

def triangulate3DPosition(M1, M2, P1, P2, Lframe, Rframe):
    cLX = int(M1["m10"] / M1["m00"])
    cLY = int(M1["m01"] / M1["m00"]) 
    cRX = int(M2["m10"] / M2["m00"])
    cRY = int(M2["m01"] / M2["m00"])
    
    print "Center"
    print cLX
    print cLY        
    
    cv2.circle(frameL, (cLX, cLY), 1, (0, 0, 0), 3)    
    cv2.circle(frameR, (cRX, cRY), 1, (0, 0, 0), 3)
    
    contL = np.array((cLX, cLY), dtype=np.float).reshape(-1, 2)
    contR = np.array((cRX, cRY), dtype=np.float).reshape(-1, 2)
    
    dronePos = cv2.triangulatePoints(P1, P2, contL[0], contR[0])
      
    coords3D = []       
    #(X,Y,Z)=(x/w,y/w,z/w)
    coords3D.append(dronePos[0] / dronePos[3]) #X
    coords3D.append(dronePos[1] / dronePos[3]) #Y
    coords3D.append(dronePos[2] / dronePos[3]) #Z
    #Calc distance from camera Square then root
    hyp = math.sqrt((coords3D[0] ** 2) + (coords3D[1] ** 2))
    string = "X:{0} Y:{1} Z:{2}".format(coords3D[0], coords3D[1], coords3D[2])
    print "Calc 3D pos"
    print string
    
    cv2.putText(frameL, string, (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1, 255)
    cv2.putText(frameL, "Distance:{0}mm".format(int(hyp)), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1, 255)
    
    cv2.putText(frameR, string, (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1, 255)
    cv2.putText(frameR, "Distance:{0}mm".format(int(hyp)), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1, 255)
    
    return frameL, frameR, coords3D

vidStreamL = cv2.VideoCapture(0)
vidStreamR = cv2.VideoCapture(1)

vidStreamL.set(cv2.CAP_PROP_EXPOSURE, .01)
vidStreamR.set(cv2.CAP_PROP_EXPOSURE, .01)

"""
vidStreamL = cv2.VideoCapture("outputL.avi")
vidStreamR = cv2.VideoCapture("outputR.avi")
"""
#"""
#Recording code
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outL = cv2.VideoWriter('outputL.avi', fourcc, 20.0, (640,480))
outR = cv2.VideoWriter('outputR.avi', fourcc, 20.0, (640,480))
#"""

if sys.argv[1] == "Y":
    CalibFile = "MainCalibrationOG.txt"
    StereoCalibrate.stereoCalabrate(vidStreamL, vidStreamR, CalibFile, True)
elif sys.argv[1] == "N":
    CalibFile = "MainCalibrationSmall.txt"
    #StereoCalibrate.stereoCalabrate(vidStreamL, vidStreamR, CalibFile, False)
elif sys.argv[1] == "Y2":
    CalibFile = "MainCalabrationSecondPair.txt"
    StereoCalibrate.stereoCalabrate(vidStreamL, vidStreamR, CalibFile, True)
else:
    CalibFile = "MainCalibrationOG.txt"

#Which cameras connected calabrates the data


#size, camMatrixL, distCoeffL, camMatrixR, distCoeffR, R, T, E, F, H = StereoCalibrate.importFile(CalibFile)
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
"""
print "Projection Matrix"
print P1
print P2

print "Rotation"
print R1
print R2

print "Transformation Matrix"
print Q

"""

cv2.namedWindow("imageL")
cv2.moveWindow("imageL", 0, 0)
cv2.namedWindow("imageR")
cv2.moveWindow("imageR", 1000, 0)

#BackSubreaction
#GMGbs, kernel = StereoBackSub.defGMGBackSub()

while(1):
  
    retL, frameL = vidStreamL.read()
    retR, frameR = vidStreamR.read()

    if (retL == False or retR == False):
        print "Frames empty"
        break
    
    frameGrayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    frameGrayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    
    #unDistImgL = cv2.undistort(frameGrayL, camMatrix[0], distCoeff[0])
    #unDistImgR = cv2.undistort(frameGrayR, camMatrix[1], distCoeff[1])      
    
    #bothFound, droneContL, noNoiseL, droneContR, noNoiseR = StereoBackSub.stereoFindDrone(frameGrayL, frameGrayR, GMGbs, kernel, camMatrix, distCoeff)
    #bothFound, droneContL, droneContR, otherCodesL, otherCodesR = QRCode.findDrone(frameGrayL, frameGrayR)
    droneContL, droneContR = Glyph.findGlyph(frameGrayL, frameGrayR)   
    
    coords3D = []
    #print H[0][0]
    if len(droneContL) > 0 and len(droneContR) > 0:#bothFound == True:
        """ Calculate disparity map"""
        
        #print droneContL
        #print droneContL[1]
        #disparity = stereo.compute(frameGrayL, frameGrayR)        
        #cv2.imshow("Disparity", disparity)
        #print("DroneContours")
        """        
        if len(droneContL) > 0 and len(droneContR) > 0:        
            Ltemp = np.array(droneContL[1])
            Rtemp = np.array(droneContR[1])    
        else:
            Ltemp = np.array(otherCodesL[1])
            Rtemp = np.array(otherCodesR[1])        
        """
        #print droneContL[1][1]    
        #print droneContR[1] 
        
        #M1 = cv2.moments(Ltemp)
        #M2 = cv2.moments(Rtemp)

        M1 = cv2.moments(droneContL)
        M2 = cv2.moments(droneContR)
                
        #protect against division by zero
        if (M1["m00"] is not 0) and (M2["m00"] is not 0): 
            frameL, frameR, coords3D = triangulate3DPosition(M1, M2, P1, P2, frameL, frameR)                    
        else:
            print "Div by zero fallthrough/protection triggered"
    
        #calcPosPlane = (dronePos[0]/dronePos[2], dronePos[1]/dronePos[2])
        #print(calcPosPlane)
        #cv2.drawContours(frameL, [Ltemp], -1, (0, 255, 255), 3)    
        #cv2.drawContours(frameR, [Rtemp], -1, (0, 255, 255), 3)   

        cv2.drawContours(frameL, [droneContL], -1, (0, 255, 255), 3)    
        cv2.drawContours(frameR, [droneContR], -1, (0, 255, 255), 3)   

        #cv2.circle(frameL, (coords3D, coords3D), -1, (0, 0, 255), 3)
        #cv2.circle(frameR, calcPosPlane, -1, (0, 0, 255), 3)
    else:
        print "LeftCam: {0}, RightCam: {1}".format(len(droneContL), len(droneContR))
        #time.sleep(.4)

    print "3D position {0}".format(coords3D)

    #cv2.imshow("noNoiseL", noNoiseL)
    #cv2.imshow("noNoiseR", noNoiseR)
    cv2.imshow("imageL", frameL)
    cv2.imshow("imageR", frameR)  

    #"""
    #Recording Code
    outL.write(frameL)
    outR.write(frameR)
    #"""
    
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
#"""
#Recording Code
outL.release()
outR.release()
#"""
vidStreamL.release()    
vidStreamR.release()
cv2.destroyAllWindows()
