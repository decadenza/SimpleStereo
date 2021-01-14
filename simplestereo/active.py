"""
active
======
Contains different active stereo algorithms and relative utilities.
"""
import os

import numpy as np
import cv2
from scipy.interpolate import interp1d

def generateGrayCodeImgs(targetDir, resolution, addHorizontal=True):
    """
    Generate Gray Codes and save it to PNG images.
    
    Starts from the couple of images *0.png* and *1.png* (one is the inverse of the other) containing vertical stripes.
    Then 2.png is coupled with 3.png and so on.
    The function stores also a *black.png* and *white.png* images for threshold calibration.
    
    Parameters
    ----------
    targetDir : string
        Path to the directory where to save Gray codes. Directory is created if not exists.
    resolution : tuple
        Pixel dimensions of the images as (width, height) tuple (to be matched with projector resolution).
    addHorizontal : bool, optional
        Whether append also horizontal patterns after the vertical ones. Default to True.
    
    Returns
    -------
    int
        Number of generated patterns (black and white are *not* considered in this count).
        If `addHorizontal` is True, the first half contains vertical stripes, followed by horizontal ones.
    """
    width, height = resolution
    graycode = cv2.structured_light_GrayCodePattern.create(width, height)
    
    num_patterns = graycode.getNumberOfPatternImages() # Surely a even number
    
    if not addHorizontal:
        num_patterns = int(num_patterns/2) # Consider vertical stripes only (first half)
    
    # Generate patterns    
    exp_patterns = graycode.generate()[1]
    
    # Create dir if not exists
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    
    # Save images
    for i in range(num_patterns):
        cv2.imwrite(os.path.join(targetDir, str(i) + '.png'), exp_patterns[i])
    
    # Additionally save black and white images (not counted in return value)
    cv2.imwrite( os.path.join(targetDir,'black.png'), (np.zeros((height, width), np.uint8)) )      # black
    cv2.imwrite( os.path.join(targetDir,'white.png'), (np.full((height, width), 255, np.uint8)) )  # white
    
    return num_patterns


def buildFringe(period=10, dims=(1280,720), color=(0,0,255), dtype=np.uint8, horizontal=False):
    """
    Build discrete sinusoidal fringe image.
    
    Parameters
    ----------
    period : float
        Fringe period along x axis, in pixels.
    dims : tuple
        Image dimensions as (width, height).
    color : tuple
        BGR color for the central stripe. If none, no stripe is drawn and
        a grayscale image is returned. Default to red (0,0,255).
    dtype: numpy.dtype
        Image is scaled in the range 0 - max value to match `dtype`.
        Default np.uint8 (max 255).
    horizontal : bool
        If True, the fringe is done with horizontal stripes.
        Default to False (vertical stripes).
        
    Returns
    -------
    numpy.ndarray
        Fringe image.
    """
    
    if horizontal:
        
        col = np.iinfo(dtype).max * ( (1 + np.sin(2*np.pi*(1/period)*np.arange(dims[1], dtype=float)))/2 )[:,np.newaxis]
        
        if color is not None:
            col = np.repeat(col[:, :, np.newaxis], 3, axis=2)
            top = int( period * ( int(dims[1]/(2*period)) - 0.25 ) )
            bottom = int(top+period)
            col[top:bottom, 0, 0] *= color[0]/255
            col[top:bottom, 0, 1] *= color[1]/255 
            col[top:bottom, 0, 2] *= color[2]/255 
            
        fullFringe = np.repeat(col.astype(dtype), dims[0], axis=1)
        
    else:
        
        row = np.iinfo(dtype).max * ((1 + np.sin(2*np.pi*(1/period)*np.arange(dims[0], dtype=float)))/2)[np.newaxis,:]
        
        if color is not None:
            row = np.repeat(row[:, :, np.newaxis], 3, axis=2)
            left = int( period * ( int(dims[0]/(2*period)) - 0.25 ) )
            right = int(left+period)
            row[0, left:right, 0] *= color[0]/255
            row[0, left:right, 1] *= color[1]/255 
            row[0, left:right, 2] *= color[2]/255 
            
        fullFringe = np.repeat(row.astype(dtype), dims[1], axis=0)
        
    return fullFringe


    
def findCentralStripe(fringe, color, threshold=100, horizontal=False):
    """
    Find coordinates of a colored stripe in the image.
    
    Search is done with subpixel accuracy only along the
    fringe front direction.
    
    Parameters
    ----------
    fringe : numpy.ndarray
        BGR image with colored stripe.
    color : tuple or list
        BGR color of the original stripe.
    threshold : int
        Threshold for color matching in 0-255 range.
    horizontal : bool
        Fringe orientation. Default to False (vertical fringe).
    
    Returns
    -------
    numpy.ndarray
        x,y coordinates of stripe centers with shape (n,2). 
    
    Notes
    -----
    The search is done along a single dimension.
    Missing values are filled with nearest-value interpolation.
    """
    h,w = fringe.shape[:2]
    maxValue = np.iinfo(fringe.dtype).max
    
    lower_color_bounds = np.array([max((c-threshold),0)*maxValue/255 for c in color])
    upper_color_bounds = np.array([min(c+threshold,255)*maxValue/255 for c in color])
    mask = cv2.inRange(fringe,lower_color_bounds,upper_color_bounds)
    fringe = cv2.cvtColor(fringe,cv2.COLOR_BGR2GRAY)
    fringe = fringe & mask
    
    def getCenters(img, axis=0):
        n = img.shape[axis]
        s = [1] * img.ndim
        s[axis] = -1
        i = np.arange(n).reshape(s)
        return np.sum(img * i, axis=axis) / np.sum(img, axis=axis)
    
    if horizontal:
        y = getCenters(fringe,axis=0)
        x = np.arange(0.5,w,1)  # Consider pixel center, as first is in x=0.5
        #res = np.hstack((x,y)).T              # x,y coordinates
        #res = res[~np.isnan(res).any(axis=1)] # Remove rows with NaN
        f = interp1d(x,y,kind="nearest",fill_value="extrapolate") # Interpolate
        y = f(x)
        
    else:
        x = getCenters(fringe,axis=1)
        y = np.arange(0.5,h,1)                # Consider pixel center, as first is in y=0.5
        #res = np.vstack((x,y)).T              # x,y coordinates
        #res = res[~np.isnan(res).any(axis=1)] # Remove rows with NaN
        f = interp1d(y,x,kind="nearest",fill_value="extrapolate") # Interpolate
        x = f(y)
    
    return np.vstack((x, y)).T
