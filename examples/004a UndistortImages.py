import sys
import os

import cv2

import simplestereo as ss
"""
Remove lens distortion from images using StereoRig
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
imgPath = os.path.join(curPath,"res","2", "calib")
loadFile = os.path.join(curPath,"res","2","rig.json")      # StereoRig file

# Load stereo rig from file
rig = ss.StereoRig.fromFile(loadFile)

# Read right and left image (please ensure the order!!!)
img1 = cv2.imread(os.path.join(imgPath,'28_L.png'))
img2 = cv2.imread(os.path.join(imgPath,'28_R.png'))


# Simply undistort two images
img1, img2 = rig.undistortImages(img1, img2)

# Show images
cv2.imshow('img1 Undistorted', img1)
cv2.imshow('img2 Undistorted', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done!")
