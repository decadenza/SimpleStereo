import sys
import os

import numpy as np
import cv2

import simplestereo as ss
"""
Rectify a couple of images using a RectifiedStereoRig
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
imgPath = os.path.join(curPath,"res","1")
loadFile = os.path.join(curPath,"res","1","rigRect.json")      # StereoRig file

# Load stereo rig from file
rigRect = ss.RectifiedStereoRig.fromFile(loadFile)

# Read right and left image (please ensure the order!!!)
img1 = cv2.imread(os.path.join(imgPath,'left.png'))
img2 = cv2.imread(os.path.join(imgPath,'right.png'))

# Optional
rigRect.computeRectificationMaps(alpha=0) # Alpha=0 may help removing unwanted border

# Simply rectify two images (it takes care of distortion too)
img1_rect, img2_rect = rigRect.rectifyImages(img1, img2)

# Show images together
visImg = np.hstack((img1_rect, img2_rect))

# Draw some horizontal lines as reference
# (after rectification all horizontal lines are epipolar lines)
for y in [289, 332, 362]:
    cv2.line(visImg, (0,y), (visImg.shape[1],y), color=(0,0,255), thickness=2)

cv2.imshow('Rectified images', visImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
