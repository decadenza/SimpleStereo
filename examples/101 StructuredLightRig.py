import sys
import os

import cv2
import numpy as np

import simplestereo as ss
"""
Load stereo rig as StructuredLightRig and triangulate some points.
"""

# Paths
curPath = os.path.dirname(os.path.realpath(__file__))
loadFile = os.path.join(curPath,"res","2","rig.json")      # StereoRig file

# Load stereo rig from file
rig = ss.StereoRig.fromFile(loadFile)

rig = ss.StructuredLightRig(rig)


# DEMO ONLY
# YOU HAVE TO FIND CORRESPONDENCES with your favourite SL algorithm
pc = np.array([[50.1,5], [10,3]])
pp = np.array([[50,5.2], [10.2,3]])

print("3D points are:")
print(rig.triangulate(pc, pp))


print("Done!")
