import sys
import os

import numpy as np
import cv2

import simplestereo as ss
"""
Use OpenCV basic matching algorithm on rectified images.
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
imgPath = os.path.join(curPath,"res","1")
loadFile = os.path.join(curPath,"res","1","rigRect.json")      # StereoRig file

# Load stereo rig from file
rigRect = ss.RectifiedStereoRig.fromFile(loadFile)

# Read right and left image (please ensure the order!!!)
img1 = cv2.imread(os.path.join(imgPath,'test_L.png'))
img2 = cv2.imread(os.path.join(imgPath,'test_R.png'))

# Simply rectify two images
img1_rect, img2_rect = rigRect.rectifyImages(img1, img2)

# Call OpenCV passive stereo algorithms
# Quality is really low due to absence of texture
stereo = cv2.StereoSGBM_create(minDisparity=10, numDisparities=85, blockSize=11)
# In this case, disparity will be multiplied by 16 internally! Divide by 16 to get real value.
disparityMap = stereo.compute(img1_rect, img2_rect).astype(np.float32)/16

# Normalize and apply a color map
disparityImg = cv2.normalize(src=disparityMap, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)

# Show only left image as reference
cv2.imshow('LEFT Rectified', img1_rect)
cv2.imshow('RIGHT Rectified', img2_rect)
cv2.imshow("Disparity ColorMap", disparityImg)

# Press ESC to close
while True:
    if cv2.waitKey(0) == 27:
        break
cv2.destroyAllWindows()
