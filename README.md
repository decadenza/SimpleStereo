# SimpleStereo
Stereo vision made simple.

SimpleStereo aims to be a support framework for stereo vision applications.

## Dependencies
* Python 3
* NumPy
* SciPy
* OpenCV

## General features
* StereoRig and RectifiedStereoRig classes to easily manage your stereo rigs
* Basic stereo capture using OpenCv `cv2.videoCapture`
* Passive stereo calibration (chessboard)
* Camera-Projector calibration (adapted from https://github.com/kamino410/procam-calibration)
* Export raw point cloud to PLY file
* The StereoFTP algorithm (improvement of Fourier Transform Profilometry) 

## Stereo rectification algorithms
- [x] Fusiello et al.
- [x] Loop and Zhang
- [x] Lafiosca and Ceccaroni

## Stereo Matching algorithms
- [x] Adaptive Support Weight algorithm
- [x] Geodesic Support Weight algorithm (*original Mutual Information implementation not completed*)


## Basic example
SimpleStereo helps you with common tasks.

```python
import simplestereo as ss
calibrationImages = ["0.png","1.png","2.png",...]
# Calibrate and build StereoRig object
rig = ss.calibration.chessboardStereo( images, chessboardSize=(7,6), squareSize=60.5 )
# Save rig object to file
rig.save("myRig.json")
# Optionally print some info
print("Reprojection error:", rig.reprojectionError)
print("Centers:", rig.getCenters())
print("Baseline:", rig.getBaseline())
```
    
More examples available.
