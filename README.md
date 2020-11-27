# SimpleStereo
Stereo vision made simple. With Python 3.

SimpleStereo aims to be a support framework for stereo vision applications.

## Dependencies
* NumPy
* SciPy
* OpenCV

## General features
* StereoRig and RectifiedStereoRig classes to easily manage your stereo rigs
* Basic stereo capture using OpenCv `cv2.videoCapture`
* Passive stereo calibration (chessboard)
* Camera-Projector calibration (adapted from https://github.com/kamino410/procam-calibration)
* Export raw point cloud to PLY file

## Stereo rectification algorithms
- [x] Fusiello
- [x] Loop and Zhang
- [x] Lafiosca and Ceccaroni

## Stereo Matching algorithms
- [x] Adaptive Support Weight algorithm
- [ ] Geodesic Support Weight algorithm with Mutual Information (to be completed)


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
