# SimpleStereo
Stereo vision made Simple.

SimpleStereo is a high level framework for stereo vision applications. It is written in Python 3, with C++ extensions.
Documentation is available at https://decadenza.github.io/SimpleStereo/

## Dependencies
* Python 3 (tested with 3.9.2)
* NumPy
* SciPy
* OpenCV
* matplotlib (for data visualisation)

## Installation

Before starting, be sure to have the latest `setuptools` package by running `pip install --upgrade setuptools`. Then proceed with one of the two options below.

### Option 1
Install package from [PyPI](https://pypi.org/project/SimpleStereo/) with:
```
pip3 install simplestereo
```

### Option 2
Clone or download the latest version and unzip. Then, from the root folder (the one containing `pyproject.toml`), run:
```
pip3 install .
```
### Troubleshooting
I am aware of some issues while installing SimpleStereo. If you have errors during installation, please open an issue.

#### Windows users troubleshooting

If during installation you get, together with other messages, the following error:
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```
Please install the Microsoft C++ Build Tools as indicated. These are required to build C++ extensions that are part of SimpleStereo.
More information about compiling on Windows is available at [https://wiki.python.org/moin/WindowsCompilers](https://wiki.python.org/moin/WindowsCompilers).

## Basic example
SimpleStereo helps you with common tasks. You can calibrate two cameras and initialise a `stereoRig` with:

```python
import simplestereo as ss

# Path to your images
images = [
    ("0_left.png", "0_right"),
    ("1_left.png", "1_right"),
    ("2_left.png", "2_right"),
    ...
    ]

# Calibrate and build StereoRig object
rig = ss.calibration.chessboardStereo( images, chessboardSize=(7,6), squareSize=60.5 )

# Save rig object to file
rig.save("myRig.json")

# Optionally print some info
print("Reprojection error:", rig.reprojectionError)
print("Centers:", rig.getCenters())
print("Baseline:", rig.getBaseline())
```
    
More examples available in the [examples](https://github.com/decadenza/SimpleStereo/tree/master/examples) folder.

## Features

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
- [x] StereoFTP (Lafiosca P. et al., [Automated Aircraft Dent Inspection via a Modified Fourier Transform Profilometry Algorithm](https://doi.org/10.3390/s22020433), Sensors, 2022)

### Unwrapping algorithms
- [x] Infinite impulse response (Estrada et al., [Noise robust linear dynamic system for phase unwrapping and smoothing](https://doi.org/10.1364/OE.19.005126), Optics Express, 2011) 

## Documentation
Documentation follows [numpydoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html).

Install documentation prerequisites with:
```
pip install Sphinx numpydoc
```

Build documentation with:
```
cd sphinx-documentation-generator
sh BUILD_SCRIPT.sh
cd ..
```

## Deploy
After building the documentation and pulled changes in master branch, assign a version in `pyproject.toml`. Tag the commit accordingly.

Then, build `*.tar.gz` distribution package:
```
python3 -m build --sdist
```
Test upload on PyPI test repository:
```
python3 -m twine upload --repository testpypi dist/*
```
Finally, upload to PyPI officially repository:
```
python3 -m twine upload dist/*
```

Optionally, clean up the `dist` folder with:
```
rm -r dist
```

## Future work
- Fix distortion coefficient issue (OpenCV related) when using 12 coefficients (currently 0, 4, 5, 8 and 14 are supported).
- Add support for fisheye cameras.
- Adapt structured light algorithms to work with two cameras.
- ArUco camera calibration algorithm.

## Contributions
Reporting issues and proposing integrations of other stereo vision algorithms is highly encouraged and it will be acknowledged.
Please share your issues!
