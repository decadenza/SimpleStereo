"""
calibration
===========
Contains different calibration algorithms.

.. todo::
    Implement circles calibration. N.B. after using ``cv2.findCirclesGrid()`` a point refinement algorithm is needed (like  ``cv2.cornerSubPix()`` does for the chessboard).
"""
import os
import warnings

import numpy as np
import cv2
from scipy.ndimage import map_coordinates

import simplestereo as ss


# Constants definitions
DEFAULT_CHESSBOARD_SIZE = (6,7) # As inner (rows, columns)
DEFAULT_CORNERSUBPIX_WINSIZE = (11,11)
DEFAULT_TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)


def chessboardSingle(images, chessboardSize = DEFAULT_CHESSBOARD_SIZE, squareSize=1, showImages=False):
    """
    Calibrate a single camera with a chessboard pattern.
    
    Parameters
    ----------
    images : list or tuple       
        A list (or tuple) of image paths, e.g. ["one.png", "two.png", ...]
    chessboardSize: tuple
        Chessboard *internal* dimensions as (width, height). Dimensions should be different to avoid ambiguity.
        Default to (7,6).
    squareSize : float
        If the square size is known, calibration can be in metric units. Default to 1.
    showImages : bool
        If True, each processed image is showed to check for correct chessboard detection.
        Default to False.
    
    Returns
    -------
    retval : bool
        Same values of `cv2.calibrateCamera`.
    cameraMatrix : numpy.ndarray
    distCoeffs : numpy.ndarray
    rvecs : numpy.ndarray
    tvecs : numpy.ndarray
    """
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),...
    objp = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * squareSize
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize)

        # If found, add object points and image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, DEFAULT_CORNERSUBPIX_WINSIZE, (-1,-1), DEFAULT_TERMINATION_CRITERIA)
            imgpoints.append(corners2)
            if showImages:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, chessboardSize, corners2,ret)
                cv2.imshow('Chessboard',img)
                cv2.waitKey(0)
        
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    
        
def chessboardStereo(images, chessboardSize = DEFAULT_CHESSBOARD_SIZE, squareSize=1):
    """
    Performs stereo calibration between two cameras and returns a StereoRig object.
    
    First camera (generally left) will be put in world origin.
    
    Parameters
    ----------
    images : list or tuple       
        A list (or tuple) of 2 dimensional tuples (ordered left and
        right) of image paths, e.g. [("oneL.png","oneR.png"),
        ("twoL.png","twoR.png"), ...]
    chessboardSize: tuple
        Chessboard *internal* dimensions as (width, height). Dimensions should be different to avoid ambiguity.
        Default to (7,6).
    squareSize : float
        If the square size is known, calibration can be in metric units. Default to 1.
        
    Returns
    ----------
    StereoRig
        A StereoRig object
    
    ..todo::
        Iteratively exclude images that have high reprojection errors and re-calibrate.
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
    retval, intrinsic1, distCoeffs1, intrinsic2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate( np.array([[objp]] * counter), imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize = img1.shape[::-1], flags=flags, criteria = DEFAULT_TERMINATION_CRITERIA)
    
    # Build StereoRig object
    stereoRigObj = ss.StereoRig(img1.shape[::-1][:2], img2.shape[::-1][:2], intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F = F, E = E, reprojectionError = retval)
    
    return stereoRigObj


def chessboardProCam(images, projectorResolution, chessboardSize = DEFAULT_CHESSBOARD_SIZE, squareSize=1, 
                     black_thr=40, white_thr=5, camIntrinsic=None, camDistCoeffs=None):
    """
    Performs stereo calibration between a camera (reference) and a projector.
    
    Adapted from the code available (MIT licence) at https://github.com/kamino410/procam-calibration
    and based on the paper of Daniel Moreno and Gabriel Taubin, "Simple, accurate, and
    robust projector-camera calibration", DOI: 10.1109/3DIMPVT.2012.77.
    The camera will be put in world origin.
    
    Parameters
    ----------
    images : list or tuple       
        A list of lists (one per set) of image paths acquired by the camera.
        Each set must be ordered like all the Gray code patterns (see ``cv2.structured_light_GrayCodePattern``)
        followed by black, normal light and white images (in this order).
        At least 5-6 sets are suggested.
    projectorResolution: tuple
        Projector pixel resolution as (width, height).
    chessboardSize: tuple, optional
        Chessboard *internal* dimensions as (width, height). Dimensions should be different to avoid ambiguity.
        Default to (7,6).
    squareSize : float, optional
        If the square size is known, calibration can be in metric units. Default to 1.
    black_thr : int, optional
       Black threshold is a number between 0-255 that represents the minimum brightness difference
       required for valid pixels, between the fully illuminated (white) and the not illuminated images (black).
       Default to 40.
    white_thr : int, optional
        White threshold is a number between 0-255 that represents the minimum brightness difference
        required for valid pixels, between the graycode pattern and its inverse images. Default to 5.
    camIntrinsic : numpy.ndarray, optional
        A 3x3 matrix representing camera intrinsic parameters. If not given it will be calculated.
    camIntrinsic : list, optional
        Camera distortion coefficients of 4, 5, 8, 12 or 14 elements (refer to OpenCV documentation).
        If not given they will be calculated.
    normalize : bool
        If True, the images are min-max normalized before processing. Default to False.
                    
    Returns
    ----------
    StereoRig
        A StereoRig object
    
    .. todo::
        Iteratively exclude images that have high reprojection errors and re-calibrate.
    """
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),...
    objps = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objps[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * squareSize
    
    
    # Gray Code setup
    gc_width, gc_height = projectorResolution
    graycode = cv2.structured_light_GrayCodePattern.create(width=gc_width, height=gc_height)
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)
    
    cam_shape = cv2.imread(images[0][0], cv2.IMREAD_GRAYSCALE).shape
    patch_size_half = int(np.ceil(cam_shape[1] / 180))
    
    cam_corners_list = []
    cam_objps_list = []
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []
    
    
    skipped = 0 # Skipped corners
    
    # Iterate over sets of Gray code images
    for imageset in images:
        
        # Check that the input images are the right number
        if len(imageset) != graycode.getNumberOfPatternImages() + 3:
            raise ValueError(f'Invalid number of images in set {os.path.dirname(imageset[0])}!')
             
        imgs = []
        for fname in imageset:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if cam_shape != img.shape:
                raise ValueError(f'Image size of {fname} is mismatch!')
            
            
            imgs.append(img)
        
        white_img = imgs.pop()
        normal_img = imgs.pop()
        black_img = imgs.pop()
        
        res, cam_corners = cv2.findChessboardCorners(normal_img, chessboardSize)
        
        if not res:
            raise ValueError(f'Chessboard not found in set {os.path.dirname(imageset[0])}!')
        
        # Subpixel refinement
        cam_corners_sub = cv2.cornerSubPix(normal_img, cam_corners, DEFAULT_CORNERSUBPIX_WINSIZE, (-1,-1), DEFAULT_TERMINATION_CRITERIA)
        
        cam_corners_list.append(cam_corners_sub)    
        cam_objps_list.append(objps)
        
        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        for corner, objp in zip(cam_corners, objps):
            c_x = int(round(corner[0][0]))
            c_y = int(round(corner[0][1]))
            src_points = []
            dst_points = []
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy
                    
                    err, proj_pix = graycode.getProjPixel(imgs, x, y)
                    
                    if not err:
                        src_points.append((x, y))
                        dst_points.append(np.array(proj_pix))
                    
            if len(src_points) < patch_size_half**2:
                #warnings.warn(f"Corner {c_x},{c_y} was skipped because decoded pixel were too few.")
                skipped+=1
                continue
                
            h_mat, inliers = cv2.findHomography(
                np.array(src_points), np.array(dst_points))
            point = h_mat@np.array([corner[0][0], corner[0][1], 1]).transpose()
            point_pix = point[0:2]/point[2]
            proj_objps.append(objp)
            proj_corners.append([point_pix])
            cam_corners2.append(corner)
            
        if len(proj_corners) < 3:
            raise ValueError(f"Not enough corners were found in set {os.path.dirname(imageset[0])} (less than 3). Skipping.")
            
            
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))
        
    if skipped>0:
        warnings.warn(f"{skipped} over {len(proj_objps_list)*chessboardSize[0]*chessboardSize[1]} skipped corners.")
    
    #print('Initial solution of camera\'s intrinsic parameters')
    cam_rvecs = []
    cam_tvecs = []
    
    # Calibrate camera only if intrinsic parameters are not given
    if camIntrinsic is None:
        cam_rms, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
            cam_objps_list, cam_corners_list, cam_shape[::-1], None, None, None, None)
    else:
        for objp, corners in zip(cam_objps_list, cam_corners_list):
            cam_rms, cam_rvec, cam_tvec = cv2.solvePnP(objp, corners, camIntrinsic, camDistCoeffs) 
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
        cam_int = camIntrinsic
        cam_dist = camDistCoeffs
    
    # Calibrate projector
    proj_rms, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, projectorResolution, None, None, None, None)
    
    # Stereo calibrate
    retval, intrinsic1, distCoeffs1, intrinsic2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        proj_objps_list, cam_corners_list2, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None, flags=cv2.CALIB_FIX_INTRINSIC)
    
    # Build StereoRig object
    stereoRigObj = ss.StereoRig(cam_shape[::-1], projectorResolution, intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F = F, E = E, reprojectionError = retval)
    
    return stereoRigObj


def _getWhiteCenters(cam_corners_list, cam_int, cam_dist, chessboardSize, squareSize):
    """
    From ordered camera chessboard corners and camera intrinsics, get
    world coordinate of the center of the white squares.
    
    These areas are less affected by ambiguity and noise due to the absence of
    high contrast pixel values.
    """
    
    # Index of corners situated at the upper-left of white squares 
    whiteUpperLeftIndexes = []
    
    for i in np.arange(1,chessboardSize[0]*(chessboardSize[1]-1)-1,2):
        sel = i
        r = (i+1)//chessboardSize[0]
        if r%2==1 and chessboardSize[0]%2==0:
            sel+=1
        # Skip end of row
        if (sel+1)%chessboardSize[0]==0:
            continue   
        whiteUpperLeftIndexes.append(sel)
    
    # Prepare white object points, like (1,0,0), (3,0,0), (5,0,0)
    whiteObjps = np.zeros((len(whiteUpperLeftIndexes),3), dtype=np.float32)
    for i,w in enumerate(whiteUpperLeftIndexes):
        whiteObjps[i,0] = (w//chessboardSize[0])*squareSize
        whiteObjps[i,1] = (w%chessboardSize[0])*squareSize
    
    # Not necessary, as origin is arbitrary
    #whiteObjps[:,:2] += squareSize/2 # Shift wrt origin along x and y
    
    cam_whiteCorners_list = []
    
    # UNDISTORT CHESSBOARDS
    for i,points in enumerate(cam_corners_list):
        # Undistort points
        points_undist = cv2.undistortPoints(points, cam_int, cam_dist)
        whiteCenters_undist = []
        # Given the upper left of valid white squares, find center
        for w in whiteUpperLeftIndexes:
            # As intersection of diagonals
            xa,ya = points_undist[w,0]
            xb,yb = points_undist[w+1,0]
            xd,yd = points_undist[w+chessboardSize[0],0]
            xc,yc = points_undist[w+chessboardSize[0]+1,0]
            
            xCenter = (xb*(yd-yb)*(xc-xa)+(ya-yb)*(xd-xb)*(xc-xa) - xa*(yc-ya)*(xd-xb)) / ((yd-yb)*(xc-xa)-(yc-ya)*(xd-xb))
            yCenter = (yc-ya)*(xCenter-xa)/(xc-xa) + ya
            
            whiteCenters_undist.append([(xCenter,yCenter)])
        
        whiteCenters_dist = ss.points.distortPoints(whiteCenters_undist,cam_dist)
            
        # Assign back camera intrinsics
        whiteCenters = cv2.perspectiveTransform(whiteCenters_dist,cam_int)   
        cam_whiteCorners_list.append(whiteCenters.astype(np.float32)) # List of [[x,y]]
    
    return cam_whiteCorners_list, whiteObjps
    
    
def chessboardProCamWhite(images, projectorResolution, chessboardSize = DEFAULT_CHESSBOARD_SIZE, squareSize=1, 
                     black_thr=40, white_thr=5, camIntrinsic=None, camDistCoeffs=None, extended=False):
    """
    Performs stereo calibration between a camera (reference) and a projector.
    
    Requires a chessboard with *black* top-left square.
    Adapted from the code available (MIT licence) at https://github.com/kamino410/procam-calibration
    and based on the paper of Daniel Moreno and Gabriel Taubin, "Simple, accurate, and
    robust projector-camera calibration", DOI: 10.1109/3DIMPVT.2012.77.
    The camera will be put in world origin.
    
    Parameters
    ----------
    images : list or tuple       
        A list of lists (one per set) of image paths acquired by the camera.
        Each set must be ordered like all the Gray code patterns (see ``cv2.structured_light_GrayCodePattern``)
        followed by black, normal light conditions and white images (in this exact order).
        At least 5-6 sets are suggested.
    projectorResolution: tuple
        Projector pixel resolution as (width, height).
    chessboardSize: tuple, optional
        Chessboard *internal* dimensions as (cols, rows). Dimensions should be different to avoid ambiguity.
        Default to (6,7).
    squareSize : float, optional
        If the square size is known, calibration can be in metric units. Default to 1.
    black_thr : int, optional
       Black threshold is a number between 0-255 that represents the minimum brightness
       difference required for valid pixels, between the fully illuminated (white) and
       the not illuminated images (black); used in computeShadowMasks method.
       Default to 40.
    white_thr : int, optional
        White threshold is a number between 0-255 that represents the minimum brightness difference
        required for valid pixels, between the graycode pattern and its inverse images; used in
        getProjPixel method.
        Default to 5.
    camIntrinsic : numpy.ndarray, optional
        A 3x3 matrix representing camera intrinsic parameters. If not given it will be calculated.
    camIntrinsic : list, optional
        Camera distortion coefficients of 4, 5, 8, 12 or 14 elements (refer to OpenCV documentation).
        If not given they will be calculated.
    extended: bool
        If True, `perViewErrors` of the stereo calibration are returned. Default to False.
        
    Returns
    -------
    StereoRig
        A StereoRig object
    perViewErrors
        Returned only if `extended` is True. For each pattern view, the
        RMS error of camera and projector is returned.
    """
    
    # Prepare camera object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),...
    objps = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objps[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * squareSize
    
    
    # Gray Code setup
    graycode = cv2.structured_light_GrayCodePattern.create(width=projectorResolution[0], height=projectorResolution[1])
    
    # CALCULATE black_threshold and white_threshold as in paper
    # TODO
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)
    
    cam_shape = cv2.imread(images[0][0], cv2.IMREAD_GRAYSCALE).shape
    patch_size_half = int(np.ceil(cam_shape[1] / 180))
    
    
    cam_corners_list = []
    cam_objps_list = []
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []
    
    
    skipped = 0 # Skipped corners
    
    #### Camera calibration
    cam_rvecs = []
    cam_tvecs = []
    
    # Calibrate camera only if intrinsic parameters are not given
    
    for imageset in images:
        
        # Load only the normal light image
        normal_img = cv2.imread(imageset[-2], cv2.IMREAD_GRAYSCALE)
        
        # Find chessboard corners
        res, cam_corners = cv2.findChessboardCorners(normal_img, chessboardSize)
        
        if not res:
            raise ValueError(f'Chessboard not found in {imageset[-2]}!')
        
        # Subpixel refinement
        cam_corners_sub = cv2.cornerSubPix(normal_img, cam_corners, DEFAULT_CORNERSUBPIX_WINSIZE, (-1,-1), DEFAULT_TERMINATION_CRITERIA)
        
        # Draw and display the corners
        #img = cv2.drawChessboardCorners(normal_img, chessboardSize, cam_corners_sub, True)
        #cv2.imshow('chessboard',img)
        #cv2.waitKey(0)
        
        cam_corners_list.append(cam_corners_sub)
        cam_objps_list.append(objps)
        
    # Do camera calibration if needed
    if camIntrinsic is None:
        # OpenCV prefers (width,height) as resolution
        cam_rms, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
            cam_objps_list, cam_corners_list, cam_shape[::-1], None, None, None, None)
    else:
        for objp, corners in zip(cam_objps_list, cam_corners_list):
            cam_rms, cam_rvec, cam_tvec = cv2.solvePnP(objp, corners, camIntrinsic, camDistCoeffs) 
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
        cam_int = camIntrinsic
        cam_dist = camDistCoeffs
    
    # From camera corners, get centers of white squares
    cam_whiteCorners_list, whiteObjps = _getWhiteCenters(cam_corners_list,
                        cam_int, cam_dist, chessboardSize, squareSize)
        
    # Iterate over sets of Gray code images
    for setnum, imageset in enumerate(images):
        
        # Check that the input images are the right number
        if len(imageset) != graycode.getNumberOfPatternImages() + 3:
            raise ValueError(f'Invalid number of images in set {os.path.dirname(imageset[0])}!')
             
        imgs = []
        for fname in imageset[:-3]: # Exclude non pattern images
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            
            if cam_shape != img.shape:
                raise ValueError(f'Image size of {fname} is mismatch!')
            imgs.append(img)
        
        
        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        for pointIndex, center in enumerate(cam_whiteCorners_list[setnum]):
            c_x = int(round(center[0][0]))
            c_y = int(round(center[0][1]))
            src_points = []
            dst_points = []
            
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy
                    
                    # Returns integer coord
                    err, proj_pix = graycode.getProjPixel(imgs, x, y)
                    
                    if not err:
                        src_points.append((x, y))
                        dst_points.append(np.array(proj_pix))
                    
            if len(src_points) < patch_size_half**2:
                #warnings.warn(f"Corner {c_x},{c_y} was skipped because decoded pixel were too few.")
                skipped+=1
                continue
            
            
            proj_objps.append(whiteObjps[pointIndex])
            
            h_mat, inliers = cv2.findHomography(
                np.array(src_points), np.array(dst_points))
            point = h_mat@np.array([center[0][0], center[0][1], 1]).transpose()
            point_pix = point[0:2]/point[2]
            proj_corners.append([point_pix])
            cam_corners2.append(center)
            
        if len(proj_corners) < 3:
            raise ValueError(f"Not enough corners were found in set {os.path.dirname(imageset[0])} (less than 3). Skipping.")
            
            
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))
        
    if skipped>0:
        warnings.warn(f"{skipped} over {len(proj_objps_list)*chessboardSize[0]*chessboardSize[1]} skipped corners.")
    
    
    # Calibrate projector
    proj_rms, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, projectorResolution, None, None, None, None)
    
    # Stereo calibrate
    retval, intrinsic1, distCoeffs1, intrinsic2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        proj_objps_list, cam_corners_list2, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None, flags=cv2.CALIB_FIX_INTRINSIC)
    
    
    # Build StereoRig object
    stereoRigObj = ss.StereoRig(cam_shape[::-1], projectorResolution, intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F = F, E = E, reprojectionError = retval)
    
    print("cam_rms", cam_rms)
    print("proj_rms", proj_rms)
    print("stereo_rms", retval)
    
    if extended:
        _, _, _, _, _, _, _, _, _, perViewErrors = cv2.stereoCalibrateExtended(
            proj_objps_list, cam_whiteCorners_list, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None, R, T, flags=cv2.CALIB_FIX_INTRINSIC)
        
        return stereoRigObj, perViewErrors
    
    else:
        return stereoRigObj


def phaseShift(periods, projectorResolution, cameraImages, chessboardSize=DEFAULT_CHESSBOARD_SIZE, squareSize=1,
               camIntrinsic=None, camDistCoeffs=None):
    """
    Calibrate camera and projector using phase shifting and the 
    heterodyne principle [Reich 1997].
    
    Parameters
    ----------
    periods : list of lists
        Periods of fringe used in the projector as list of lists.
        First list is for the horizontal fringes, second for the vertical ones.
        In descending order (e.g. 1280, 1024, 512, ...).
    projectorResolution: tuple
        Projector pixel resolution as (width, height).
    cameraImages : list or tuple       
        A list of lists (one per set) of image paths acquired by the camera.
        Each set must be ordered, having 4 images for each period with
        horizontal followed by vertical images.
        In each set, last image is the one in normal light conditions.
        At least 5-6 sets are suggested.
    chessboardSize: tuple, optional
        Chessboard *internal* dimensions as (cols, rows). Dimensions should be different to avoid ambiguity.
        Default to (6,7).
    squareSize : float, optional
        If the square size is known, calibration can be in metric units. Default to 1.
    camIntrinsic : numpy.ndarray, optional
        A 3x3 matrix representing camera intrinsic parameters. If not given it will be calculated.
    camIntrinsic : list, optional
        Camera distortion coefficients of 4, 5, 8, 12 or 14 elements (refer to OpenCV documentation).
        If not given they will be calculated.
        
    Returns
    -------
    StereoRig
        A StereoRig object
    """
    
    
    # Internal functions
    def getPhase(imgPaths):
        """
        Get ordered images and retrieve wrapped phase map.
        """
        I = []
        for p in imgPaths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            I.append(img.astype(float))
            
        # cos(theta + i*pi/2) expected  
        # Output values in [0, 2pi)
        return np.mod( np.arctan2(I[3]-I[1], I[0]-I[2]), 2*np.pi)
        
    
    def unwrap(theta0, theta1, T0, T1):
        """
        Given absolute phase `theta0`, wrapped phase `theta1` and their
        respective periods `T0` and `T1`, unwrap `theta1`.
        
        Returned phase is normalized in [0, 2pi).
        """
        k = np.rint((theta0*T0/T1 - theta1)/(2*np.pi))
        return (theta1 + 2*np.pi*k)*T1/T0
    
    
    def getProjCoord(phase_x, phase_y):
        """
        Return projector coordinate corresponding to absolute phase values.
        """
        px = projectorResolution[0]*phase_x/(2*np.pi)
        py = projectorResolution[1]*phase_y/(2*np.pi)
        return np.hstack( (px, py) )
    
    
    # Prepare camera object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),...
    objps = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objps[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * squareSize
    
    cam_shape = cv2.imread(cameraImages[0][0], cv2.IMREAD_GRAYSCALE).shape
    cam_corners_list = []
    cam_objps_list = []
    proj_corners_list = []
    proj_objps_list = []
    
    #### Camera calibration
    cam_rvecs = []
    cam_tvecs = []
    
    
    for imageset in cameraImages:
        
        # Get corners from normal.png as [(x1,y1), (x2,y2), ...]
        normal_img = cv2.imread(imageset[-1], cv2.IMREAD_GRAYSCALE)
        res, cam_corners = cv2.findChessboardCorners(normal_img, chessboardSize)
        
        if not res:
            raise ValueError(f'Chessboard not found in {imageset[-1]}!')
        
        # Subpixel refinement. Output as [(x1,y1), (x2,y2), ...]
        cam_corners_sub = cv2.cornerSubPix(normal_img, cam_corners, DEFAULT_CORNERSUBPIX_WINSIZE, (-1,-1), DEFAULT_TERMINATION_CRITERIA)
        
        # Draw and display the corners
        #img = cv2.drawChessboardCorners(normal_img, chessboardSize, cam_corners_sub, True)
        #cv2.imshow('chessboard',img)
        #cv2.waitKey(0)
        
        cam_corners_list.append(cam_corners_sub.astype(np.float32))
        cam_objps_list.append(objps.astype(np.float32))
    
    
        # Get absolute phase using all the acquired fringes
        
        i=0 # counter
        phase = [None,None] # Horizontal and vertical absolute phases
        for v in range(2):
            for j,T in enumerate(periods[v]):
                if j==0: # if it is the first (maximum period)
                    phase[v] = getPhase(imageset[i:i+4])
                else:
                    phase2 = getPhase(imageset[i:i+4])
                    phase[v] = unwrap(phase[v], phase2, periods[v][0], T)
                    
                i+=4 # update counter
        
                    
        # Find camera-projector correspondences
        #cam_corners_sub shape (n, 1, 2)
        corners = cam_corners_sub.reshape(-1,2) # shape (n, 2) [[x1,y1], [x2,y2], ...]
        corners = corners.T # (2, n)
        corners = corners[[1,0]] # Swap x,y to get
        
        # Linear interpolation of non-integer coordinates
        phase_x = map_coordinates(phase[0], corners, order=1).reshape(-1,1)
        phase_y = map_coordinates(phase[1], corners, order=1).reshape(-1,1)
        
        # Get corresponding values on projector
        proj_corners = getProjCoord(phase_x,phase_y) # Shape (n, 2)
          
        proj_corners_list.append(proj_corners.astype(np.float32))
        proj_objps_list.append(objps.astype(np.float32))
    
    
    # Call OpenCV calibrate
    # Calibrate camera only if intrinsic parameters are not given
    if camIntrinsic is None:
        cam_rms, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
            cam_objps_list, cam_corners_list, cam_shape[::-1], None, None, None, None)
    else:
        for objp, corners in zip(cam_objps_list, cam_corners_list):
            cam_rms, cam_rvec, cam_tvec = cv2.solvePnP(objp, corners, camIntrinsic, camDistCoeffs) 
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
        cam_int = camIntrinsic
        cam_dist = camDistCoeffs
    
    # Calibrate projector
    proj_rms, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, projectorResolution, None, None, None, None)
    
    # Stereo calibrate
    retval, intrinsic1, distCoeffs1, intrinsic2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        proj_objps_list, cam_corners_list, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None, flags=cv2.CALIB_FIX_INTRINSIC)
    
    # Build StereoRig object
    stereoRigObj = ss.StereoRig(cam_shape[::-1], projectorResolution, intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F = F, E = E, reprojectionError = retval)
    
    return stereoRigObj
    

def phaseShiftWhite(periods, projectorResolution, cameraImages, chessboardSize=DEFAULT_CHESSBOARD_SIZE,
                    squareSize=1, camIntrinsic=None, camDistCoeffs=None, extended=False):
    """
    Calibrate camera and projector using phase shifting and heterodyne
    principle [Reich 1997]. Using center of white squares instead of
    corners as targets.
    
    The center of a white square is well defined and less subject to
    noise or uncertanty of the phase value.
    
    Parameters
    ----------
    periods : list of lists
        Periods of fringe used in the projector as list of lists.
        First list is for the horizontal fringes, second for the vertical ones.
        In descending order (e.g. 1280, 1024, 512, ...).
    projectorResolution: tuple
        Projector pixel resolution as (width, height).
    cameraImages : list or tuple       
        A list of lists (one per set) of image paths acquired by the camera.
        Each set must be ordered, having 4 images for each period with
        horizontal followed by vertical images.
        In each set, last image is the one in normal light conditions.
        At least 5-6 sets are suggested.
    chessboardSize: tuple, optional
        Chessboard *internal* dimensions as (cols, rows). Dimensions should be different to avoid ambiguity.
        Default to (6,7).
    squareSize : float, optional
        If the square size is known, calibration can be in metric units. Default to 1.
    camIntrinsic : numpy.ndarray, optional
        A 3x3 matrix representing camera intrinsic parameters. If not given it will be calculated.
    camIntrinsic : list, optional
        Camera distortion coefficients of 4, 5, 8, 12 or 14 elements (refer to OpenCV documentation).
        If not given they will be calculated.
    extended: bool
        If True, `perViewErrors` of the stereo calibration are returned. Default to False.
        
    Returns
    -------
    StereoRig
        A StereoRig object
    perViewErrors
        Returned only if `extended` is True. For each pattern view, the
        RMS error of camera and projector is returned.
    """
    
    
    # Internal functions
    def getPhase(imgPaths):
        """
        Get ordered images and retrieve wrapped phase map.
        """
        I = []
        for p in imgPaths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            I.append(img.astype(float))
            
        # cos(theta + i*pi/2) expected  
        # Output values in [0, 2pi)
        return np.mod( np.arctan2(I[3]-I[1], I[0]-I[2]), 2*np.pi)
        
    
    def unwrap(theta0, theta1, T0, T1):
        """
        Given absolute phase `theta0`, wrapped phase `theta1` and their
        respective periods `T0` and `T1`, unwrap `theta1`.
        
        Returned phase is normalized in [0, 2pi).
        """
        k = np.rint((theta0*T0/T1 - theta1)/(2*np.pi))
        return (theta1 + 2*np.pi*k)*T1/T0
    
    
    def getProjCoord(phase_x, phase_y):
        """
        Return projector coordinate corresponding to absolute phase values.
        """
        px = projectorResolution[0]*phase_x/(2*np.pi)
        py = projectorResolution[1]*phase_y/(2*np.pi)
        return np.hstack( (px, py) )
    
    
    # Prepare camera object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),...
    objps = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objps[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * squareSize
    
    cam_shape = cv2.imread(cameraImages[0][0], cv2.IMREAD_GRAYSCALE).shape
    cam_corners_list = []
    cam_objps_list = []
    proj_corners_list = []
    proj_objps_list = []
    
    #### Camera calibration
    cam_rvecs = []
    cam_tvecs = []
    
    # Loop 1
    for imageset in cameraImages:
        
        # Get corners from normal.png as [(x1,y1), (x2,y2), ...]
        normal_img = cv2.imread(imageset[-1], cv2.IMREAD_GRAYSCALE)
        res, cam_corners = cv2.findChessboardCorners(normal_img, chessboardSize)
        
        if not res:
            raise ValueError(f'Chessboard not found in {imageset[-1]}!')
        
        # Subpixel refinement. Output as [(x1,y1), (x2,y2), ...]
        cam_corners_sub = cv2.cornerSubPix(normal_img, cam_corners, DEFAULT_CORNERSUBPIX_WINSIZE, (-1,-1), DEFAULT_TERMINATION_CRITERIA)
        
        # Draw and display the corners
        #img = cv2.drawChessboardCorners(normal_img, chessboardSize, cam_corners_sub, True)
        #cv2.imshow('chessboard',img)
        #cv2.waitKey(0)
        
        cam_corners_list.append(cam_corners_sub.astype(np.float32))
        cam_objps_list.append(objps.astype(np.float32))
    
    # Call OpenCV calibrate
    # Calibrate camera only if intrinsic parameters are not given
    if camIntrinsic is None:
        cam_rms, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
            cam_objps_list, cam_corners_list, cam_shape[::-1], None, None, None, None)
    else:
        for objp, corners in zip(cam_objps_list, cam_corners_list):
            cam_rms, cam_rvec, cam_tvec = cv2.solvePnP(objp, corners, camIntrinsic, camDistCoeffs) 
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
        cam_int = camIntrinsic
        cam_dist = camDistCoeffs
    
    
    # From camera corners, get centers of white squares
    cam_whiteCorners_list, whiteObjps = _getWhiteCenters(cam_corners_list,
                        cam_int, cam_dist, chessboardSize, squareSize)
    
    whiteObjps = whiteObjps.astype(np.float32)
    cam_whiteObjps_list = [[whiteObjps] for _ in range(len(cam_whiteCorners_list))]
    
    # Loop 2
    for setnum, imageset in enumerate(cameraImages):
        # Get absolute phase using all the acquired fringes
        i=0 # counter
        phase = [None,None] # Horizontal and vertical absolute phases
        for v in range(2):
            for j,T in enumerate(periods[v]):
                if j==0: # if it is the first (maximum period)
                    phase[v] = getPhase(imageset[i:i+4])
                else:
                    phase2 = getPhase(imageset[i:i+4])
                    phase[v] = unwrap(phase[v], phase2, periods[v][0], T)
                
                i+=4 # update counter
        
        ### Find camera-projector correspondences
        #cam_corners_sub shape (n, 1, 2)
        corners = cam_whiteCorners_list[setnum].reshape(-1,2) # shape (n, 2) [[x1,y1], [x2,y2], ...]
        corners = corners.T # (2, n)
        corners = corners[[1,0]] # Swap to get (y,x) coords
        
        # Linear interpolation of non-integer coordinates
        phase_x = map_coordinates(phase[0], corners, order=1).reshape(-1,1)
        phase_y = map_coordinates(phase[1], corners, order=1).reshape(-1,1)
        
        # Get corresponding (x,y) values on projector
        proj_corners = getProjCoord(phase_x,phase_y) # Shape (n, 2)
        
        proj_corners_list.append(proj_corners.astype(np.float32))
        proj_objps_list.append(whiteObjps)
    
    # Calibrate projector
    proj_rms, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, projectorResolution, None, None, None, None)
    '''
    # Calibrate projector WITHOUT LENS DISTORTION
    proj_rms, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, projectorResolution, None, None, None, None, flags=cv2.CALIB_FIX_K1+cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_ZERO_TANGENT_DIST)
    '''
    
    # Stereo calibrate
    retval, intrinsic1, distCoeffs1, intrinsic2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        proj_objps_list, cam_whiteCorners_list, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None, flags=cv2.CALIB_FIX_INTRINSIC)

    # Build StereoRig object
    stereoRigObj = ss.StereoRig(cam_shape[::-1], projectorResolution, intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F = F, E = E, reprojectionError = retval)
    
    #TEMP
    print("cam_rms", cam_rms)
    print("proj_rms", proj_rms)
    print("stereo_rms", retval)
    
    if extended:
        _, _, _, _, _, _, _, _, _, perViewErrors = cv2.stereoCalibrateExtended(
            proj_objps_list, cam_whiteCorners_list, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None, R, T, flags=cv2.CALIB_FIX_INTRINSIC)
        
        return stereoRigObj, perViewErrors
    
    else:
        return stereoRigObj


def generateChessboardSVG(chessboardSize, filepath, squareSize=20, border=10):
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
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{cols*squareSize}mm" height="{rows*squareSize}mm" viewBox="0 0 {cols} {rows}" style="border: {border}mm solid #FFF;">')
        f.write('<rect fill="#FFF" x="0" y="0" width="{}" height="{}"/>'.format(cols, rows))
        d = 'M0 0'
        d += 'm0 2'.join(['H{}v1H0z'.format(cols) for _ in range((rows+1)//2)]) # Build rows
        d += 'M1 0'
        d += 'm2 0'.join(['V{}h1V0z'.format(rows) for _ in range(cols//2)]) # Build cols
        f.write('<path fill="#000" d="{}"/></svg>'.format(d))
    
    return
    
        
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
