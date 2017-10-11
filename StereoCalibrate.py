
import sys
import cv2
import numpy as np
import cPickle

def stereoCalabrate(vidStreamL, vidStreamR, fileName, calabBool):
    cv2.namedWindow("grayL")
    cv2.moveWindow("grayL", 0, 0)
    cv2.namedWindow("grayR")
    cv2.moveWindow("grayR", 1000, 0)
    cv2.namedWindow("imgL")
    cv2.moveWindow("imgL", 0, 1000)
    cv2.namedWindow("imgR")
    cv2.moveWindow("imgR", 1000, 1000)

    print "SereoCalib Running"

    #vidStreamL = cv2.VideoCapture(0)
    #vidStreamR = cv2.VideoCapture(1)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Arrays to store object points and image points from all the images.
    all3DPoints = [] # 3d point in real world space
    imgpointsL = [] #[] # 2d points in image plane.
    imgpointsR = [] #[]
    counter = 0
    #SmallChessboardCalibration
    if calabBool == False:
        vertChess = 3 
        horzChess = 5
        objp = np.zeros((vertChess*horzChess,3), np.float32)
        objp[:,:2] = np.mgrid[0:vertChess,0:horzChess].T.reshape(-1,2)*38
    else:
        #OGChessboardCalibration
        vertChess = 6 
        horzChess = 9
        objp = np.zeros((vertChess*horzChess,3), np.float32)
        objp[:,:2] = np.mgrid[0:vertChess,0:horzChess].T.reshape(-1,2)*28
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)    

    
    #x,y = np.meshgrid(range(7),range(6))
    #worldPoints = np.hstack((x.reshape(42,1),y.reshape(42,1),np.zeros((42,1)))).astype(np.float32)
    
    #while(counter < 60):
    #while(counter < 40):
    while(counter < 20):     
        retL, frameL = vidStreamL.read()
        retR, frameR = vidStreamR.read()
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        #retCorL , cornersL = cv2.findChessboardCorners(grayL, (6,9),None)
        #retCorR , cornersR = cv2.findChessboardCorners(grayR, (6,9),None)
        retCorL , cornersL = cv2.findChessboardCorners(grayL, (vertChess,horzChess),None)
        retCorR , cornersR = cv2.findChessboardCorners(grayR, (vertChess,horzChess),None)
              
         
        #print (retCorL)
        cv2.imshow("grayL", grayL)
        cv2.imshow("grayR", grayR)
    
        if retCorL == True and retCorR == True:
            counter += 1
            print "Dual Images captured: {}".format(counter)        
            all3DPoints.append(objp)
            #all3DPoints.append(worldPoints)
            corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
            corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
            
                      
            imgpointsL.append(corners2L)    
            imgpointsR.append(corners2R)    
            #imgpointsR = np.append(imgpointsR, corners2R) #imgpointsR.append(corners2R)
            #imgpointsL = np.append(imgpointsL, corners2L) #imgpointsL.append(corners2L)
            
          
            
            # Draw and display the corners
            #imgR = cv2.drawChessboardCorners(frameR, (7,6), corners2R,retR)
            #imgL = cv2.drawChessboardCorners(frameL, (7,6), corners2L,retL)
            imgR = cv2.drawChessboardCorners(frameR, (vertChess,horzChess), corners2R,retR)
            imgL = cv2.drawChessboardCorners(frameL, (vertChess,horzChess), corners2L,retL)
                    
            cv2.imshow('imgR',imgR)
            cv2.imshow('imgL',imgL)        
            #cv2.waitKey(200) #imgpointsR
    
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    
    print "Starting Calibration of Cameras"
    
    #stereo = cv2.cv2.createStereoBM(numDisparities=16, blockSize=15)
    
    height, width = grayL.shape[:2]
    
    #retMonoL, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all3DPoints, imgpointsL, (width, height), None, None)
    retMonoL, mtxL, distL, _, _ = cv2.calibrateCamera(all3DPoints, imgpointsL, (height, width), None, None)
    retMonoL, mtxR, distR, _, _ = cv2.calibrateCamera(all3DPoints, imgpointsR, (height, width), None, None)
    retStereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(all3DPoints, imgpointsL, imgpointsR, mtxL,distL,mtxR,distR,(width, height))
    #print corners2L[0][0] 
    #print cornersR    
    #cv2.waitKey(0)  
    #print np.array(imgpointsL[0])
    #print imgpointsR     
    
    #print np.array(imgpointsL)
    #print np.array(imgpointsR)
    #H = cv2.getPerspectiveTransform(np.array(imgpointsL[0]), np.array(imgpointsR[0]))
    #retStereo, cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, R, T, E, F = cv2.stereoCalibrate(all3DPoints, imgpointsL, imgpointsR, mtx,dist,mtx,dist,(width, height))
    cv2.destroyAllWindows()
    #vidStreamL.release()    
    #vidStreamR.release()  
    print "Calibration Complete"
    
    """
    print("Camera Matrices Left : Right")
    print(mtxL) #Camera matrics
    print(mtxR)    
    print("Distorition Coefficents Left : Right")
    print(distL)
    print(distR)    
    print("Rotation Matrix")
    print(R) #Rotation Matrix
    print("Translation Vector")
    print(T) #Translation Vector
    print("Essential Matrix")
    print(E) #Essential Matrix
    print("Fundamental Matrix")
    print(F) #Fundamental Matrix
    print "Homography Matrix"
    print H
    """
    
    print "Exporting to txt file"
    
    #Need to add the 3rd camera to the calabration
    #data = {"camera_matrix": mtx, "dist_coeff": dist, "rotation_matrix": R, "trans_vector": T, "essential_matrix": E, "fundamental_matrix": F}
    #data = {"camera_matrix": mtx, "dist_coeff": dist}
    #fileName = "calibrationData.txt"
    
    with open(fileName, "w") as f:
        cPickle.dump((height, width), f)
        cPickle.dump(mtxL, f)
        cPickle.dump(distL, f)
        cPickle.dump(mtxR, f)
        cPickle.dump(distR, f)        
        cPickle.dump(R, f)
        cPickle.dump(T, f)
        cPickle.dump(E, f)
        cPickle.dump(F, f)
        #cPickle.dump(H, f)
        
    print "Export to {0} Complete".format(fileName)
    
def importFile(fileName):
    print "Importing {0}".format(fileName)
    
    with open(fileName, "r") as fp:
        size = cPickle.load(fp)
        camMatrixL = cPickle.load(fp)
        distCoeffL = cPickle.load(fp)
        camMatrixR = cPickle.load(fp)
        distCoeffR = cPickle.load(fp)
        rotMatrix = cPickle.load(fp)
        transVector = cPickle.load(fp)
        essentialMatrix = cPickle.load(fp)
        fundMatrix = cPickle.load(fp)
        #homoMatrix = cPickle.load(fp)
    
    print "Image size for calibration (Height, Width)"
    print "Height {0}, Width {1}".format(size[0], size[1])    
    print "Camera Matrices Left : Right"
    print camMatrixL
    print camMatrixR
    print "Distorition Coefficents Left : Right"
    print distCoeffL
    print distCoeffR
    print "Rotation Matrix"
    print rotMatrix
    print "Translation Vector"
    print transVector
    print "Essential Matrix"
    print essentialMatrix
    print "Fundamental Matrix" 
    print fundMatrix 
    #print "Homography Matrix"
    #print homoMatrix
    
    print("Import Complete")
    return size, camMatrixL, distCoeffL, camMatrixR, distCoeffR, rotMatrix, transVector, essentialMatrix, fundMatrix#, homoMatrix
    
"""
#cameraMatrix1 = np.zeros(3, 3)
#cameraMatrix2 = np.zeros(3, 3)

    

retval, cameraMatrixR, distCoeffsR, cameraMatrixL, distCoeffsL, R, T, E, F = cv2.stereoCalibrate(all3DPoints, imgpointsR, imgpointsL, (width, height), None, None)
print("Complete")
#retCalR, mtxCalR, distCalR, rvecsCalR, tvecsCalR = cv2.calibrateCamera(all3DPoints, imgpointsR, grayR.shape[::-1], None, None)
#retCalL, mtxCalL, distCalL, rvecsCalL, tvecsCalL = cv2.calibrateCamera(all3DPoints, imgpointsL, grayL.shape[::-1], None, None)

print ("Starting Rectification")
R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))
P1 = np.zeros(shape=(3,3))
P2 = np.zeros(shape=(3,3))

cv2.stereoRectify(cameraMatrixR, distCoeffsR, cameraMatrixL, distCoeffsL, (width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv2.cv.CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0) )

print "Done Rectification\n"
print "Applying Undistort\n"



map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (width, height), cv2.CV_32FC1)

print "Undistort complete\n"

while(True):
    retL, img1 = vidStreamL.read()
    retR, img2 = vidStreamR.read()
    imgU1 = np.zeros((height,width,3), np.uint8)
    imgU1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
    imgU2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    cv2.imshow("imageL", img1);
    cv2.imshow("imageR", img2);
    cv2.imshow("image1L", imgU1);
    cv2.imshow("image2R", imgU2);
    k = cv2.waitKey(5);
    if(k==27):
        break;

#print(all3DPoints)
"""