import sys
import cv2
import numpy as np
import StereoCalibrate

StereoCalibrate.stereoCalabrate("MainCalabration.txt")

StereoCalibrate.importFile("MainCalabration.txt")