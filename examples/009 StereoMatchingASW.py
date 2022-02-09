"""
Rectify a couple of images and use ASW stereo matching algorithm.
"""

import sys
import os

import numpy as np
import cv2

import simplestereo as ss


# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
imgPath = os.path.join(curPath,"res","2")
loadFile = os.path.join(curPath,"res","2","rigRect.json")      # StereoRig file

# Load stereo rig from file
rigRect = ss.RectifiedStereoRig.fromFile(loadFile)

# Read right and left image (please ensure the order!!!)
img1 = cv2.imread(os.path.join(imgPath,'lawn_L.png'))
img2 = cv2.imread(os.path.join(imgPath,'lawn_R.png'))

# Simply rectify two images
img1_rect, img2_rect = rigRect.rectifyImages(img1, img2)

# WARNING: This algorithm is EXTREMELY SLOW.
# So let's resize to speed up the matching process.
img1_rect = cv2.resize(img1_rect, None, fx=0.25, fy=0.25)
img2_rect = cv2.resize(img2_rect, None, fx=0.25, fy=0.25)

stereo = ss.passive.StereoASW(winSize=35, minDisparity=4, maxDisparity=25, gammaC=15, gammaP=17.5, consistent=False)

# Get disparity map
# Returned disparity is unsigned int 16 bit.
disparityMap = stereo.compute(img1_rect, img2_rect)

# Normalize and apply a color map
disparityImg = cv2.normalize(src=disparityMap, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)

# Show only left image as reference
cv2.imshow('LEFT Rectified', img1_rect)
cv2.imshow('RIGHT Rectified', img2_rect)
cv2.imshow("Disparity ColorMap", disparityImg)
cv2.waitKey(0)

cv2.destroyAllWindows()
