# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:20:25 2017

@author: william
"""
import cv2
import zbar
import numpy as np

def findDrone(imageL, imageR):
    LdroneInfo, RdroneInfo, LotherCodes, RotherCodes = [], [], [], []
    bothFound = False
    # create a reader
    scanner = zbar.Scanner()

    resultsL = scanner.scan(imageL)
    resultsR = scanner.scan(imageR)

    #stringBuilder = ""
   
    if len(resultsL) > 0 and len(resultsR) > 0:
        bothFound = True
        for L in resultsL:
            if (L.data == "Left") or (L.data == "Right") or (L.data == "Front") or (L.data == "Back"):
                LdroneInfo.append(L.data)
                LdroneInfo.append(L.position)
            else:
                LotherCodes.append(L.data)
                LotherCodes.append(L.position)
        for R in resultsR:
            if (R.data == "Left") or (R.data == "Right") or (R.data == "Front") or (R.data == "Back"):
                RdroneInfo.append(R.data)
                RdroneInfo.append(R.position)
            else:
                RotherCodes.append(R.data)
                RotherCodes.append(R.position)
    elif len(resultsL) > 0:
        for L in resultsL:
            if (L.data == "Left") or (L.data == "Right") or (L.data == "Front") or (L.data == "Back"):
                LdroneInfo.append(L.data)
                LdroneInfo.append(L.position)
            else:
                LotherCodes.append(L.data)
                LotherCodes.append(L.position)
    elif len(resultsR) > 0:
        for R in resultsR:
            if (R.data == "Left") or (R.data == "Right") or (R.data == "Front") or (R.data == "Back"):
                RdroneInfo.append(R.data)
                RdroneInfo.append(R.position)
            else:
                RotherCodes.append(R.data)
                RotherCodes.append(R.position)
    #else:
        #print "QR not found"            
    return bothFound, LdroneInfo, RdroneInfo, LotherCodes, RotherCodes

"""    
    elif len(resultsL) > 0 or len(resultsR) > 0:
        print "Single Found"
    elif stringBuilder == "":
        print "No QR Code"
"""
    #print stringBuilder#"{0}:{1}".format(result.data, result.position)
        

#pil = Image.fromarray(gray)
#width, height = pil.size
"""
raw = pil.tostring()


# wrap image data
image = zbar.Image(width, height, 'Y800', raw)

# scan the image for barcodes
scanner.scan(image)

# extract results
for symbol in image:
    # do something useful with results
    if symbol.data == "None":
        print "Drone bevindt zich buiten het raster"
    else:
        print symbol.data
"""