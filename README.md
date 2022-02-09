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

## Features and future work

### General
* `StereoRig`, `RectifiedStereoRig` and `StructuredLightRig` classes to easily manage your stereo rigs and their calibration
* Basic stereo capture using OpenCV `cv2.videoCapture`
* Export and import point cloud to PLY file
 
### Calibration algorithms
- [x] Chessboard calibration (one and two cameras)
- [x] Camera-projector calibration adapted (Moreno D. et al.), adapted from [procam](https://github.com/kamino410/procam-calibration) (`ss.calibration.chessboardProCam`)
- [x] Camera-projector calibration alternative version (`ss.calibration.chessboardProCamWhite`)

### Stereo rectification algorithms
- [x] Fusiello et al. (`ss.rectification.fusielloRectify`)
- [x] Wrapper of OpenCV algorithm (`ss.rectification.stereoRectify`)
- [x] Loop and Zhang (`ss.rectification.loopRectify`)
- [x] Lafiosca and Ceccaroni (`ss.rectification.directRectify`, see also [DirectStereoRectification](https://github.com/decadenza/DirectStereoRectification))

### Passive stereo matching algorithms
- [x] Adaptive Support Weight algorithm (K. Yoon et al.)
- [x] Geodesic Support Weight algorithm (*simplified implementation* from Asmaa Hosni et al.)

### Active and Structured light algorithms
- [x] Gray code
- [ ] De Brujin
- [ ] Adapt structured light algorithms to work with two cameras
- [x] StereoFTP (Lafiosca P. et al. [see full article and citation here](https://doi.org/10.3390/s22020433))

### Unwrapping algorithms
- [x] Infinite impulse response (Estrada et al.) 

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
    
More advanced examples available in the [example](https://github.com/decadenza/SimpleStereo/tree/master/examples) folder.
