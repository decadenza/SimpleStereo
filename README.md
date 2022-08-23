# SimpleStereo
Stereo Vision Made Simple.

SimpleStereo aims at being a high level framework for stereo vision applications. It is written in Python 3, with C++ extensions.
Documentation is available at https://decadenza.github.io/SimpleStereo/

## Dependencies
* Python 3 (tested with 3.9.2)
* NumPy
* SciPy
* OpenCV
* matplotlib (for data visualisation purposes)

## Installation
### Option 1
Install package from PyPI:
```
pip3 install simplestereo
```

### Option 2
Download the latest version and unzip. Then, from the folder containing `setup.py`, run:
```
pip3 install .
```

## Basic example
SimpleStereo helps you with common tasks. You can calibrate a single camera as:

```python
import simplestereo as ss

# Path to your images
images = ["0.png","1.png","2.png",...]

# Calibrate and build StereoRig object
rig = ss.calibration.chessboardStereo(images, chessboardSize=(7,6), squareSize=60.5)

# Save rig object to file
rig.save("myRig.json")

# Optionally print some info
print("Reprojection error:", rig.reprojectionError)
print("Centers:", rig.getCenters())
print("Baseline:", rig.getBaseline())
```
    
More advanced examples available in the [examples](https://github.com/decadenza/SimpleStereo/tree/master/examples) folder.

## Features and Future Work

### General
* `StereoRig`, `RectifiedStereoRig` and `StructuredLightRig` classes to easily manage your stereo rigs and their calibration
* Basic stereo capture using OpenCV `cv2.videoCapture`
* Export and import point cloud to PLY file
 
### Calibration algorithms
- [x] Chessboard calibration (one and two cameras)
- [x] Camera-projector calibration adapted (Moreno D. et al.), adapted from [procam](https://github.com/kamino410/procam-calibration) (`ss.calibration.chessboardProCam`)
- [x] Camera-projector calibration alternative version (`ss.calibration.chessboardProCamWhite`)

### Stereo rectification algorithms
- [x] Fusiello et al., *A compact algorithm for rectification of stereo pairs*, 2000 (`ss.rectification.fusielloRectify`)
- [x] Wrapper of OpenCV algorithm (`ss.rectification.stereoRectify`)
- [x] Loop and Zhang, *Computing rectifying homographies for stereo vision*, 1999 (`ss.rectification.loopRectify`)
- [x] Lafiosca and Ceccaroni, *Rectifying homographies for stereo vision: analytical solution for minimal distortion*, 2022, https://doi.org/10.1007/978-3-031-10464-0_33 (`ss.rectification.directRectify`, see also [DirectStereoRectification](https://github.com/decadenza/DirectStereoRectification))

### Passive stereo matching algorithms
- [x] Adaptive Support Weight algorithm (K. Yoon et al., *Adaptive support-weight approach for correspondence search*, 2006)
- [x] Geodesic Support Weight algorithm (*simplified implementation*, credits Asmaa Hosni et al.)

### Active and Structured light algorithms
- [x] Gray code
- [ ] De Brujin
- [ ] Adapt structured light algorithms to work with two cameras
- [x] StereoFTP (Lafiosca P. et al., [Automated Aircraft Dent Inspection via a Modified Fourier Transform Profilometry Algorithm](https://doi.org/10.3390/s22020433), Sensors, 2022)

### Unwrapping algorithms
- [x] Infinite impulse response (Estrada et al., [Noise robust linear dynamic system for phase unwrapping and smoothing](https://doi.org/10.1364/OE.19.005126), Optics Express, 2011) 

## Contributions
Reporting issues and proposing integrations of other stereo vision algorithms is highly encouraged and it will be acknowledged.
Please share your thoughts!
