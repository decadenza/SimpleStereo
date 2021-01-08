"""
calibration
===========
Contains different calibration algorithms.

.. todo::
    - Implement circles calibration. N.B. after using ``cv2.findCirclesGrid()`` a point refinement algorithm is needed (like  ``cv2.cornerSubPix()`` does for the chessboard).
"""
import os
import warnings

import numpy as np
import cv2

import simplestereo as ss

# Constants definition
DEFAULT_CHESSBOARD_SIZE = (6,7) # As inner (rows, columns)
DEFAULT_CORNERSUBPIX_WINSIZE = (11,11)
DEFAULT_TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

def chessboardStereo(images, chessboardSize = DEFAULT_CHESSBOARD_SIZE, squareSize=1):
    """
    Performs stereo calibration between two cameras and returns a StereoRig object.
    
    First camera (generally left) will be put in world origin.
    
    Parameters
    ----------
    images : list or tuple       
        A list (or tuple) of 2 dimensional tuples (ordered left and right) of image paths, e.g. [("oneL.png","oneR.png"), ("twoL.png","twoR.png"), ...]
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
    
    ..todo::
        Iteratively exclude images that have high reprojection errors and re-calibrate.
    """
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),...
    objps = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objps[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * squareSize
    
    
    # Gray Code setup
    gc_width, gc_height = projectorResolution
    graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
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
            cam_objps_list, cam_corners_list, cam_shape, None, None, None, None)
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
    stereoRigObj = ss.StereoRig(cam_shape[::-1], projectorResolution[::-1], intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F = F, E = E, reprojectionError = retval)
    
    return stereoRigObj

    
def chessboardProCamWhite(images, projectorResolution, chessboardSize = DEFAULT_CHESSBOARD_SIZE, squareSize=1, 
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
                
    Returns
    ----------
    StereoRig
        A StereoRig object
    """
    
    # Prepare camera object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),...
    objps = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objps[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * squareSize
    
    
    # Gray Code setup
    graycode = cv2.structured_light_GrayCodePattern.create(projectorResolution[0], projectorResolution[1])
    
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
        cam_rms, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
            cam_objps_list, cam_corners_list, cam_shape, None, None, None, None)
    else:
        for objp, corners in zip(cam_objps_list, cam_corners_list):
            cam_rms, cam_rvec, cam_tvec = cv2.solvePnP(objp, corners, camIntrinsic, camDistCoeffs) 
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
        cam_int = camIntrinsic
        cam_dist = camDistCoeffs
    
    # FIND MIDDLE-WHITE POINTS INDEXES
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
        whiteObjps[i,0] = w//chessboardSize[0]*squareSize
        whiteObjps[i,1] = w%chessboardSize[0]*squareSize
    
    
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
        
        whiteCenters_dist = ss.utils.distortPoints(whiteCenters_undist,cam_dist)
            
        # Assign back camera intrinsics
        whiteCenters = cv2.perspectiveTransform(whiteCenters_dist,cam_int)   
        cam_whiteCorners_list.append(whiteCenters) # List of [[x,y]]
        
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
    stereoRigObj = ss.StereoRig(cam_shape[::-1], projectorResolution[::-1], intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F = F, E = E, reprojectionError = retval)
    
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
