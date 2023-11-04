import sys
import os

import cv2

import simplestereo as ss
"""
Calibrate a single camera from chessboards.
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
loadPath = os.path.join(curPath,"res","1","calib")    # Image folder

# Total number of images
N_IMAGES = 5

# Image paths
images = [ os.path.join(loadPath,str(i)+'_L.png') for i in range(N_IMAGES) ]
print(f"Calibrating using {len(images)} images from:\n{loadPath}...")

# Calibrate
retval, cameraMatrix, distCoeffs, rvecs, tvecs = ss.calibration.chessboardSingle(images, chessboardSize=(7,6), squareSize=60.5)

# Print some info
print("Camera matrix: ", cameraMatrix)
print("Distortion coefficients: ", distCoeffs)

print("Done!")
