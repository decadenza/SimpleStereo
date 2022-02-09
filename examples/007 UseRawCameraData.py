import sys
import os

import numpy as np
import cv2

import simplestereo as ss
"""
Use raw camera parameters to build a stereo rig.
You need intrinsics and extrinsics parameteres, as well as distortion
coefficients, if available.
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
imgPath = os.path.join(curPath,"res","0")
saveFile = os.path.join(curPath,"res","0","rigRect.json")      # StereoRig file

# Read right and left image (please ensure the order!!!)
img1 = cv2.imread(os.path.join(imgPath,'left.png'))
img2 = cv2.imread(os.path.join(imgPath,'right.png'))

# Raw camera parameters (an NumPy arrays)

# Left intrinsics
A1 = np.array([[ 960,   0, 960/2], 
               [   0, 960, 540/2],
               [   0,   0,     1]]) 

# Right intrinsics        
A2 = np.array([[ 960,   0, 960/2],
               [   0, 960, 540/2],
               [   0,   0,     1]]) 

# Left extrinsics
RT1 = np.array([[ 0.98920029, -0.11784191, -0.08715574,  2.26296163],
                [-0.1284277 , -0.41030705, -0.90285909,  0.15825593],
                [ 0.07063401,  0.90430164, -0.42101002, 11.0683527 ]])

# Right extrinsics
RT2 = np.array([[ 0.94090474,  0.33686835,  0.03489951,  1.0174818 ],
                [ 0.14616159, -0.31095025, -0.93912017,  2.36511779],
                [-0.30550784,  0.88872361, -0.34181178, 14.08488464]])

# Distortion coefficients
distCoeffs1 = None
distCoeffs2 = None

# As a convention, the world origin must be in the left camera.
# Move the world origin into the first camera (IMPORTANT)
R, T = ss.utils.moveExtrinsicOriginToFirstCamera(RT1[:,:3], RT2[:,:3], RT1[:,3], RT2[:,3])

# Create the StereoRig
rig = ss.StereoRig(img1.shape[::-1][1:], img2.shape[::-1][1:], A1, A2, distCoeffs1, distCoeffs2, R, T) 

# Build the RectifiedStereoRig
rigRect = ss.rectification.directRectify(rig)

# Save it to file
rigRect.save(saveFile)

# Rectify the images
img1_rect, img2_rect = rigRect.rectifyImages(img1, img2)

# Show images
cv2.imshow('LEFT rectified', img1_rect)
cv2.imshow('RIGHT rectified', img2_rect)
cv2.waitKey(0)
cv2.destroyAllWindows()
