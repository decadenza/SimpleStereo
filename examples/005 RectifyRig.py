import sys
import os

import cv2

import simplestereo as ss
"""
Rectify a StereoRig previously created!
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
loadFile = os.path.join(curPath,"res","1","rig.json")      # StereoRig file
saveFile = os.path.join(curPath,"res","1","rigRect.json")  # Destination

# Load stereo rig from file
rig = ss.StereoRig.fromFile(loadFile)

# Choose a rectification algorithm and rectify the stereo rig
# Resulting in a RectifiedStereoRig object
rigRect = ss.rectification.directRectify(rig)       # The best one (minimum distortion)
#rigRect = ss.rectification.loopRectify(rig)
#rigRect = ss.rectification.fusielloRectify(rig)
#rigRect = ss.rectification.stereoRectify(rig)      # OpenCV standard

# Save it to file
rigRect.save(saveFile)

print("Done!")
