"""
Full test of matching and 3D point extraction from a couple of raw stereo images.
Output: disparity map + point cloud (PLY).

NOTE: Be sure to have enough texture or passive algorithms are likely to fail!
"""

import sys
import os

import numpy as np
import cv2
from scipy.ndimage import median_filter 
from scipy.signal.signaltools import wiener

import simplestereo as ss




# Read right and left image (please mantain the order as it was in calibration!!!)
img1 = cv2.imread('res/2/lawn_L.png')
img2 = cv2.imread('res/2/lawn_R.png')

# Load rectified stereo rig from file
rigRect = ss.RectifiedStereoRig.fromFile('res/2/rigRect.json')

# Optionally rectification maps can be changed in final resolution and zoom
#rigRect.computeRectificationMaps(zoom=1)
#rigRect.computeRectificationMaps((128*3,72*3), zoom=1)
# Otherwise original resolution is used

# Load non-rectified stereo rig and compute rectification 
#rig = ss.StereoRig.fromFile('examples/2/rig.json')
#rigRect = ss.rectification.directRectify(rig)

# Simply rectify two images
img1_rect, img2_rect = rigRect.rectifyImages(img1, img2)
 
#cv2.imshow('LEFT after', img1_rect)
#cv2.imshow('RIGHT after', img2_rect)
#cv2.waitKey(0)

### Call OpenCV passive stereo algorithms...
# NB Final disparity will be multiplied by 16 internally! Divide by 16 to get real value.
stereo = cv2.StereoSGBM_create(minDisparity=20, numDisparities=80, blockSize=11, uniquenessRatio=0,P1=50,P2=20)
disparityMap = stereo.compute(img1_rect, img2_rect).astype(np.float32)/16 # disparityMap coming from Stereo_SGBM is multiplied by 16

# ALTERNATIVE
# Call other SimpleStereo algorithms (much slower)
#stereo = ss.passive.StereoASW(winSize=35, minDisparity=40, maxDisparity=90, gammaC=20, gammaP=17.5, consistent=False)
#stereo = ss.passive.StereoASW(winSize=35, minDisparity=10, maxDisparity=30, gammaC=20, gammaP=17.5, consistent=False)
# Get disparity map
# Returned disparity is unsigned int 16 bit.
#disparityMap = stereo.compute(img1_rect, img2_rect) 

# Get 3D points
points3D = rigRect.get3DPoints(disparityMap)
ss.points.exportPLY(points3D, "export.ply", img1_rect)

# Normalize and color
disparityImg = cv2.normalize(src=disparityMap, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)
cv2.imwrite("disparity.png", disparityImg)

# Show only left image as reference
cv2.imshow('LEFT rectified', img1_rect)
cv2.imshow('RIGHT rectified', img2_rect)
cv2.imshow("Disparity Color", disparityImg)

print("Press ESC to close.")
while True:
    if cv2.waitKey(0) == 27:
        break
cv2.destroyAllWindows()

        
        
