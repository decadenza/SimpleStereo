# SimpleStereo
Stereo vision made simple.

SimpleStereo is a higher level framework for stereo vision applications. It is written in Python 3, with C++ extensions.

## Dependencies
* Python 3
* NumPy
* SciPy
* OpenCV
* matplotlib (for data visualisation purposes)

## Installation
Download the last version and unzip. Then, in the folder containing `setup.py`, run:
```
pip3 install .
```

*PyPi package coming soon...*

## Features

### General
* StereoRig and RectifiedStereoRig classes to easily manage your stereo rigs and their geometry
* Basic stereo capture using OpenCV `cv2.videoCapture`
* Export raw point cloud to PLY file
 
### Calibration algorithms
- [x] Chessboard calibration (single and double cameras)
- [x] Camera-projector calibration adapted from [procam](https://github.com/kamino410/procam-calibration) (see `ss.calibration.chessboardProCam` and a derived version `ss.calibration.chessboardProCamWhite`)

### Stereo rectification algorithms
- [x] Fusiello et al.
- [x] Loop and Zhang
- [x] Lafiosca and Ceccaroni

### Passive stereo matching algorithms
- [x] Adaptive Support Weight algorithm
- [x] Geodesic Support Weight algorithm (*simplified implementation*)

## Structured light algorithms
- [x] Gray code
- [ ] De Brujin (*coming soon*)
- [x] StereoFTP (Fourier transform profilometry)

### Active stereo algorithms
- [ ] Adapt structured light algorithms to work with two cameras (*coming soon*)

## Basic example
SimpleStereo helps you with common tasks. You can calibrate a single camera as:

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
    
More advanced examples available.
