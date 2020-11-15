"""
active
======
Contains different active stereo algorithms and relative utilities.
"""
import os

import numpy as np
import cv2


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
        Number of generated patterns (black and white are not considered in this count).
        First half will contain vertical stripes, followed by horizontal ones.
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
