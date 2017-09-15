
import sys
import cv2
import numpy as np
import cPickle

def calabrate(fileName):
    cv2.namedWindow("grayL")
    cv2.moveWindow("grayL", 0, 0)
    cv2.namedWindow("grayR")
    cv2.moveWindow("grayR", 1000, 0)
    cv2.namedWindow("imgL")
    cv2.moveWindow("imgL", 0, 1000)
    cv2.namedWindow("imgR")
    cv2.moveWindow("imgR", 1000, 1000)
    
    
    print("SereoCalib Running")
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    
    
    # Arrays to store object points and image points from all the images.
    all3DPoints = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    imgpointsR = []
    vidStreamL = cv2.VideoCapture(0)
    vidStreamR = cv2.VideoCapture(1)
    counter = 0
    vertChess = 6 
    horzChess = 9
    
    
    #x,y = np.meshgrid(range(7),range(6))
    #worldPoints = np.hstack((x.reshape(42,1),y.reshape(42,1),np.zeros((42,1)))).astype(np.float32)
    
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
        #width1, height1 = grayL.shape[:2]
        #width2, height2 = grayR.shape[:2]
        #print(width1)
        #print(width2)
        #print(height1)
        #print(height2)    
    
    
        if retCorL == True and retCorR == True:
            counter += 1
            print("Dual Images captured: {}").format(counter)        
            all3DPoints.append(objp)
            #all3DPoints.append(worldPoints)
            corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
            corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
            imgpointsR.append(corners2R)
            imgpointsL.append(corners2L)
            # Draw and display the corners
            #imgR = cv2.drawChessboardCorners(frameR, (7,6), corners2R,retR)
            #imgL = cv2.drawChessboardCorners(frameL, (7,6), corners2L,retL)
            imgR = cv2.drawChessboardCorners(frameR, (vertChess,horzChess), corners2R,retR)
            imgL = cv2.drawChessboardCorners(frameL, (vertChess,horzChess), corners2L,retL)
                    
            cv2.imshow('imgR',imgR)
            cv2.imshow('imgL',imgL)        
            #cv2.waitKey(200)
            
    
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    
    print("Starting Calibration of Cameras")
    
    #stereo = cv2.cv2.createStereoBM(numDisparities=16, blockSize=15)
    
    width, height = grayL.shape[:2]
    
    #retMonoL, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all3DPoints, imgpointsL, (width, height), None, None)
    retMonoL, mtx, dist, _, _ = cv2.calibrateCamera(all3DPoints, imgpointsL, (width, height), None, None)
    retStereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(all3DPoints, imgpointsL, imgpointsR, mtx,dist,mtx,dist,(width, height))
    #retStereo, cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, R, T, E, F = cv2.stereoCalibrate(all3DPoints, imgpointsL, imgpointsR, mtx,dist,mtx,dist,(width, height))
    
    print("Calibration Complete")
    """
    print("Camera Matrices")
    print(mtx) #Camera matrics
    print("Distorition Coefficents")
    print(dist)
    print("Rotation Matrix")
    print(R) #Rotation Matrix
    print("Translation Vector")
    print(T) #Translation Vector
    print("Essential Matrix")
    print(E) #Essential Matrix
    print("Fundamental Matrix")
    print(F) #Fundamental Matrix
    """
    print("Exporting to txt file")
    
    #Need to add the 3rd camera to the calabration
    #data = {"camera_matrix": mtx, "dist_coeff": dist, "rotation_matrix": R, "trans_vector": T, "essential_matrix": E, "fundamental_matrix": F}
    #data = {"camera_matrix": mtx, "dist_coeff": dist}
    #fileName = "calibrationData.txt"
    
    
    with open(fileName, "w") as f:
        cPickle.dump(mtx, f)
        cPickle.dump(dist, f)
        cPickle.dump(R, f)
        cPickle.dump(T, f)
        cPickle.dump(E, f)
        cPickle.dump(F, f)
        
    print("Export to {} Complete").format(fileName)

def importFile(fileName):
    print("Importing {}").format(fileName)
    
    with open(fileName, "r") as fp:
        camMatrix = cPickle.load(fp)
        distCoeff = cPickle.load(fp)
        rotMatrix = cPickle.load(fp)
        transVector = cPickle.load(fp)
        essentialMatrix = cPickle.load(fp)
        fundMatrix = cPickle.load(fp)
        
    print("Camera Matrices")
    print(camMatrix)
    print("Distorition Coefficents")
    print(distCoeff)
    print("Rotation Matrix")
    print(rotMatrix)
    print("Translation Vector")
    print(transVector)
    print("Essential Matrix")
    print(essentialMatrix)
    print("Fundamental Matrix")
    print(fundMatrix)
    
    print("Import Complete")
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