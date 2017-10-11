# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:53:31 2017

@author: william
"""

import cv2
import numpy as np
from glyphfunctions import *
#from glyphdatabase import *

QUADRILATERAL_POINTS = 4
SHAPE_RESIZE = 100.0
BLACK_THRESHOLD = 100
WHITE_THRESHOLD = 155

"""
GLYPH_TABLE_9Block = [[[[0, 1, 0, 1, 0, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 1, 0, 0]], "devil"],
                [[[1, 0, 0, 0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 0]], "devil_red"]]
                
"""

GLYPH_TABLE_9Block = [[[[1, 1, 1, 0, 1, 0, 0, 0, 0]], "Front"],[[[0, 1, 0, 0, 1, 0, 1, 1, 1]], "Back"],[[[1, 0, 0, 1, 1, 0, 0, 1, 0]], "Left"],[[[1, 0, 0, 0, 1, 1, 0, 1, 1]], "Right"],[[[1, 0, 1, 0, 1, 0, 1, 0, 1]], "CommonPoint"]]
GLYPH_TABLE_6Block = [[[[1, 0, 0, 1, 0, 1]], "Front"],[[[1, 1, 0, 1, 0, 0]], "Back"],[[[1, 1, 0, 0, 1, 1]], "Left"],[[[1, 0, 1, 1, 0, 1]], "Right"],[[[1, 1, 1, 1, 1, 1]], "CommonPoint"]]


# Match glyph pattern to database record
def glyphDatabase(glyph_pattern):

 
    glyph_found = False
    glyph_rotation = None
    glyph_type = None
     
    #for glyph_record in GLYPH_TABLE_9Block:
    for glyph_record in GLYPH_TABLE_6Block:
        for idx, val in enumerate(glyph_record[0]):    
            if glyph_pattern == val: 
                glyph_found = True
                glyph_rotation = idx
                glyph_type = glyph_record[1]
                break
        if glyph_found: break
 
    return (glyph_found, glyph_rotation, glyph_type)

def findGlyph(imageL, imageR):
    # Stage 1: Read an image from our webcam
    #image = webcam.get_current_frame()
 
    # Stage 2: Detect edges in image
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayL = cv2.GaussianBlur(imageL, (5,5), 0)
    grayR = cv2.GaussianBlur(imageR, (5,5), 0)
    
    edgesL = cv2.Canny(grayL, 100, 200)
    edgesR = cv2.Canny(grayR, 100, 200)
 
    # Stage 3: Find contours
    _, contoursL, _ = cv2.findContours(edgesL, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contoursR, _ = cv2.findContours(edgesR, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    contoursL = sorted(contoursL, key=cv2.contourArea, reverse=True)[:10]
    contoursR = sorted(contoursR, key=cv2.contourArea, reverse=True)[:10]
        
    #LeftGlyphs, RightGlyphs = [], []
    droneContL, droneContR = [], []
    
    for contour in contoursL:
        # Stage 4: Shape check
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)
 
        if len(approx) == QUADRILATERAL_POINTS:
 
            # Stage 5: Perspective warping
            topdown_quad = get_topdown_quad(grayL, approx.reshape(4, 2))
            #print topdown_quad[0][0]
            #print int(topdown_quad.shape[0]/100.0)*5
            #print int(topdown_quad.shape[1]/100.0)*5
            #print "{0} : {1}".format(int(topdown_quad.shape[0]/100.0)*5, int(topdown_quad.shape[1]/100.0)*5)          
            # Stage 6: Border check
            if topdown_quad[int((topdown_quad.shape[0]/100.0)*5)][int((topdown_quad.shape[1]/100.0)*5)] > BLACK_THRESHOLD: continue
 
            # Stage 7: Glyph pattern
            glyph_pattern = get_glyph_pattern(topdown_quad, BLACK_THRESHOLD, WHITE_THRESHOLD)
            glyph_found, glyph_rotation, glyph_type = glyphDatabase(glyph_pattern)
 
            if glyph_found:
 
                if glyph_type == "CommonPoint":
                    #print "Common Point Found Leftside"
                    droneContL = contour
                else:
                    droneContL = contour
                 
                #for _ in range(glyph_rotation):
                #    substitute_image = rotate_image(substitute_image, 90)
                 
                #image = add_substitute_quad(image, substitute_image, approx.reshape(4, 2))

    for contour in contoursR:
# Stage 4: Shape check
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)
 
        if len(approx) == QUADRILATERAL_POINTS:
 
            # Stage 5: Perspective warping
            topdown_quad = get_topdown_quad(grayR, approx.reshape(4, 2))
 
            if topdown_quad[int((topdown_quad.shape[0]/100.0)*5)][int((topdown_quad.shape[1]/100.0)*5)] > BLACK_THRESHOLD: continue
        
            # Stage 7: Glyph pattern
            glyph_pattern = get_glyph_pattern(topdown_quad, BLACK_THRESHOLD, WHITE_THRESHOLD)
            print "Pattern found"
            print glyph_pattern            
            glyph_found, glyph_rotation, glyph_type = glyphDatabase(glyph_pattern)
 
            if glyph_found:
                if glyph_type == "CommonPoint":
                    #print "Common Point Found Rightside"
                    droneContR = contour
                else:
                    droneContR = contour
    return droneContL, droneContR
    # Stage 9: Show augmented reality
    cv2.imshow('2D Augmented Reality using Glyphs', image)
    cv2.waitKey(10)
    
    
    
    