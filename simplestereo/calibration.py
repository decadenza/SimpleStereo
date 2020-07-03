"""
calibration
===========
Contains different calibration algorithms.

.. todo::
    - Implement circles calibration. N.B. after using ``cv2.findCirclesGrid()`` a point refinement algorithm is needed (like  ``cv2.cornerSubPix()`` does for the chessboard).
"""
import numpy as np
import cv2

import simplestereo as ss

# Constants definition
DEFAULT_CHESSBOARD_SIZE = (7,6)
DEFAULT_CORNERSUBPIX_WINSIZE = (11,11)
DEFAULT_TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

def chessboardCalibrate(images, chessboardSize = DEFAULT_CHESSBOARD_SIZE, squareSize=1):
    """
    Does stereo calibration from a list of images using OpenCV and returns a StereoRig object.
    
    Width and height of the chessboard can't be of the same length.
    
    Parameters
    ----------
    images : list or tuple       
        A list (or tuple) of 2 dimensional tuples (ordered left and right) of image paths, e.g. [("oneL.png","oneR.png"), ("twoL.png","twoR.png"), ...]
    chessboardSize: tuple
        Chessboard dimensions. Default to (7,6).
    squareSize : int or float
        If the square size is known, calibration can be in metric units. Default to 1.
        
    Returns
    ----------
    StereoRig
        A StereoRig object
        
    
    Todo
    ----
    Add a way to exclude images that have high reprojection errors and re-calibrate.
    """
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),...
    objp = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * squareSize
    
    # Arrays to store image points from all the images.
    imagePoints1 = []
    imagePoints2 = []
    
    counter = 0 # Count successful couples
    
    for path1, path2 in images:
        # Read as grayscale images
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        
        # Check that the files exist
        if img1 is None or img2 is None:
            raise ValueError("File not found!")
        
        # Find the chessboard corners
        ret1, corners1 = cv2.findChessboardCorners(img1, chessboardSize)
        ret2, corners2 = cv2.findChessboardCorners(img2, chessboardSize)
        
        if ret1 and ret2:
            # Refine the corner locations
            corners1 = cv2.cornerSubPix(img1, corners1, DEFAULT_CORNERSUBPIX_WINSIZE, (-1,-1), DEFAULT_TERMINATION_CRITERIA)
            corners2 = cv2.cornerSubPix(img2, corners2, DEFAULT_CORNERSUBPIX_WINSIZE, (-1,-1), DEFAULT_TERMINATION_CRITERIA)
            # Save to main list
            imagePoints1.append(corners1)
            imagePoints2.append(corners2)
            counter += 1
    
    # Initialize parameters
    R = np.eye(3, dtype=np.float64)             # Rotation matrix between the 1st and the 2nd camera coordinate systems.
    T = np.zeros((3, 1), dtype=np.float64)      # Translation vector between the coordinate systems of the cameras.
    cameraMatrix1 = np.eye(3, dtype=np.float64)
    cameraMatrix2 = np.eye(3, dtype=np.float64)
    distCoeffs1 = np.empty(5)
    distCoeffs2 = np.empty(5)
    
    # Flags for calibration
    # TO DO flags management to provide different configurations to user
    flags = 0
    
    # Do stereo calibration
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate( np.array([[objp]] * counter), imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize = img1.shape[::-1], flags=flags, criteria = DEFAULT_TERMINATION_CRITERIA)
    
    # Build StereoRig object
    stereoRigObj = ss.StereoRig(img1.shape[::-1][:2], img2.shape[::-1][:2], cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, F = F, E = E, reprojectionError = retval)
    
    return stereoRigObj




def generateChessboardSVG(chessboardSize, filepath, squareSize=20):
    """
    Write the desired chessboard to a SVG file.
    
    *chessboardSize* is expressed as (columns, rows) tuple, counting *internal line* columns and rows 
    as OpenCV does (e.g. to obtain a 10x7 chessboard, use (9,6)).
    
    Parameters
    ----------
    chessboardSize : tuple
        Size of the chessboard as (columns, rows).
    filepath : string
        File path where to save the SVG file.
    squareSize : int
        Side of the square in millimeters. Default to 20. However it may not be represented exactly, depending on software.
    """
    cols, rows = chessboardSize
    cols+=1
    rows+=1
    with open(filepath, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>')
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="{}mm" height="{}mm" viewBox="0 0 {} {}" style="border: {}mm solid #FFF;">'.format(cols*squareSize, rows*squareSize, cols, rows, squareSize))
        f.write('<rect fill="#FFF" x="0" y="0" width="{}" height="{}"/>'.format(cols, rows))
        d = 'M0 0'
        d += 'm0 2'.join(['H{}v1H0z'.format(cols) for _ in range((rows+1)//2)]) # Build rows
        d += 'M1 0'
        d += 'm2 0'.join(['V{}h1V0z'.format(rows) for _ in range(cols//2)]) # Build cols
        f.write('<path fill="#000" d="{}"/></svg>'.format(d))
    
    
    
def getFundamentalMatrixFromProjections(P1,P2):
    """
    Compute the fundamental matrix from two projection matrices.
    
    The algorithm is adapted from an original lesson of Cristina Turrini, UNIMI, Trento (09/04/2017).
    
    Parameters
    ----------
    P1, P2 : numpy.ndarray
        3x4 camera projection matrices.
    
    Returns
    -------
    F : numpy.ndarray
        The 3x3 fundamental matrix.
    
    """
    
    X = []
    X.append(np.vstack((P1[1,:], P1[2,:])))
    X.append(np.vstack((P1[2,:], P1[0,:])))
    X.append(np.vstack((P1[0,:], P1[1,:])))

    Y = []
    Y.append(np.vstack((P2[1,:], P2[2,:])))
    Y.append(np.vstack((P2[2,:], P2[0,:])))
    Y.append(np.vstack((P2[0,:], P2[1,:])))

    F = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            F[i, j] = np.linalg.det(np.vstack((X[j], Y[i])))
    
    return F
