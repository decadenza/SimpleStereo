import sys
import os

import cv2

import simplestereo as ss
"""
Rectify a couple of images using a RectifiedStereoRig
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
imgPath = os.path.join(curPath,"res","2")
loadFile = os.path.join(curPath,"res","2","rigRect.json")      # StereoRig file

# Load stereo rig from file
rig = ss.RectifiedStereoRig.fromFile(loadFile)

# Read right and left image (please ensure the corrrect left-right order).
img1 = cv2.imread(os.path.join(imgPath,'lawn_L.png'))
img2 = cv2.imread(os.path.join(imgPath,'lawn_R.png'))

# Simply undistort two images (if distortion is present).
img1, img2 = rig.undistortImages(img1, img2)

F = rig.getFundamentalMatrix()
ss.utils.drawCorrespondingEpipolarLines(img1, img2, F, x1=[(495,332), (726,80), (636,226)], x2=[], color=(0,0,255), thickness=3)

# Show images.
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done!")
