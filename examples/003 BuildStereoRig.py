import sys
import os

import cv2

import simplestereo as ss
"""
Build a stereo rig object calculating parameters from calibration images.
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
loadPath = os.path.join(curPath,"res","1","calib")    # Image folder
saveFile = os.path.join(curPath,"res","1","rig.json") # Destination

# Total number of images
N_IMAGES = 30

# Image paths
images = [ ( os.path.join(loadPath,str(i)+'_L.png'), os.path.join(loadPath,str(i)+'_R.png')) for i in range(N_IMAGES) ]
print(f"Calibrating using {len(images)} images from:\n{loadPath}...")

# Calibrate and build StereoRig object
rig = ss.calibration.chessboardStereo( images, chessboardSize=(7,6), squareSize=60.5 )

# Save rig object to file
rig.save( saveFile )

# Print some info
print("Saved in:", saveFile)
print("Reprojection error:", rig.reprojectionError)
print("Centers:", rig.getCenters())
print("Baseline:", rig.getBaseline())

print("Done!")
