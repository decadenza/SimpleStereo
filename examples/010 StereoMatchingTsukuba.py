import sys
import os

import numpy as np
import cv2

import simplestereo as ss
"""
Evaluate two different stereo matching algorithms on Tsukuba image pair:
- Adaptive Support Weights
- SGBM (wrapped from default OpenCV library)
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
imgPath = os.path.join(curPath,"res","tsukuba")

# Read right and left image (please ensure the order!!!)
# These images are already rectified!
img1_rect = cv2.imread(os.path.join(imgPath,'tsukuba_l.png'))
img2_rect = cv2.imread(os.path.join(imgPath,'tsukuba_r.png'))

print("Matchin in progress...")
# You can try to adjust settings of each algorithm

# Call SimpleStereo algorithms
# Using a MODIFIED version of Adaptive Support Weight algorithm from
# "Locally adaptive support-weight approach for visual correspondence search", K. Yoon, I. Kweon, 2005.
# WARNING: This algorithm is EXTREMELY SLOW on larger images.
stereoASW = ss.passive.StereoASW(winSize=35, minDisparity=4, maxDisparity=14, gammaC=15, gammaP=17.5, consistent=True)
disparityMapASW = stereoASW.compute(img1_rect, img2_rect)

# OPTIONAL Export 3D adimensional point cloud
#points = ss.points.getAdimensional3DPoints(disparityMapASW)
#ss.points.exportPLY(points, "Tsukuba.ply", img1_rect)

# Compare with OpenCV SGBM (quicker)
stereoSGBM = cv2.StereoSGBM_create(minDisparity=4, numDisparities=10, blockSize=11)
# In this case, disparity will be multiplied by 16 internally! Divide by 16 to get real value.
disparityMapSGBM = stereoSGBM.compute(img1_rect, img2_rect).astype(np.float32)/16


# Normalize and apply a color map
disparityImgASW = cv2.normalize(src=disparityMapASW, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
disparityImgASW = cv2.applyColorMap(disparityImgASW, cv2.COLORMAP_JET)
disparityImgSGBM = cv2.normalize(src=disparityMapSGBM, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
disparityImgSGBM = cv2.applyColorMap(disparityImgSGBM, cv2.COLORMAP_JET)

# Show only left image as reference
cv2.imshow('LEFT Rectified', img1_rect)
cv2.imshow('RIGHT Rectified', img2_rect)
cv2.imshow("ASW Disparity ColorMap", disparityImgASW)
cv2.imshow("SGBM Disparity ColorMap", disparityImgSGBM)

# Press ESC to close
while True:
    if cv2.waitKey(0) == 27:
        break
cv2.destroyAllWindows()
