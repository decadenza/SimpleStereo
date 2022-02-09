import sys
import os

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
img1 = cv2.imread(os.path.join(imgPath,'test_L.png'))
img2 = cv2.imread(os.path.join(imgPath,'test_R.png'))

# Simply rectify two images (it takes care of distortion too)
img1_rect, img2_rect = rigRect.rectifyImages(img1, img2)

# Draw some horizontal lines as reference (after rectification all horizontal lines are epipolar lines)
for y in [61,443]:
    cv2.line(img1_rect, (0,y), (1280,y), color=(0,0,255), thickness=1)
    cv2.line(img2_rect, (0,y), (1280,y), color=(0,0,255), thickness=1)

# Show images
cv2.imshow('img1 Rectified', img1_rect)
cv2.imshow('img2 Rectified', img2_rect)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done!")
