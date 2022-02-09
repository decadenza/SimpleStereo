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
- [x] Camera-projector calibration adapted from [procam](https://github.com/kamino410/procam-calibration) (see `ss.calibration.chessboardProCam`)
- [x] Camera-projector calibration alternative version (see `ss.calibration.chessboardProCamWhite`)

### Stereo rectification algorithms
- [x] Fusiello et al.
- [x] Loop and Zhang
- [x] Lafiosca and Ceccaroni

### Passive stereo matching algorithms
- [x] Adaptive Support Weight algorithm
- [x] Geodesic Support Weight algorithm (*simplified implementation*)

### Active and Structured light algorithms
- [x] Gray code
- [ ] De Brujin
- [ ] Adapt structured light algorithms to work with two cameras
- [x] StereoFTP (*see citation below*)

N.B. StereoFTP algorithm is discussed in detail in:

Lafiosca, Pasquale, Ip-Shing Fan, and Nicolas P. Avdelidis. "Automated Aircraft Dent Inspection via a Modified Fourier Transform Profilometry Algorithm." *Sensors* 22.2 (2022): 433.

If you find this useful, please cite as:

```
@Article{StereoFTP,
AUTHOR = {Lafiosca, Pasquale and Fan, Ip-Shing and Avdelidis, Nicolas P.},
TITLE = {Automated Aircraft Dent Inspection via a Modified Fourier Transform Profilometry Algorithm},
JOURNAL = {Sensors},
VOLUME = {22},
YEAR = {2022},
NUMBER = {2},
ARTICLE-NUMBER = {433},
URL = {https://www.mdpi.com/1424-8220/22/2/433},
PubMedID = {35062394},
ISSN = {1424-8220},
ABSTRACT = {The search for dents is a consistent part of the aircraft inspection workload. The engineer is required to find, measure, and report each dent over the aircraft skin. This process is not only hazardous, but also extremely subject to human factors and environmental conditions. This study discusses the feasibility of automated dent scanning via a single-shot triangular stereo Fourier transform algorithm, designed to be compatible with the use of an unmanned aerial vehicle. The original algorithm is modified introducing two main contributions. First, the automatic estimation of the pass-band filter removes the user interaction in the phase filtering process. Secondly, the employment of a virtual reference plane reduces unwrapping errors, leading to improved accuracy independently of the chosen unwrapping algorithm. Static experiments reached a mean absolute error of &sim;0.1&nbsp;mm at a distance of 60&nbsp;cm, while dynamic experiments showed &sim;0.3&nbsp;mm at a distance of 120&nbsp;cm. On average, the mean absolute error decreased by &sim;34%, proving the validity of the proposed single-shot 3D reconstruction algorithm and suggesting its applicability for future automated dent inspections.},
DOI = {10.3390/s22020433}
}

```

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
    
More advanced examples available in the example folder.
