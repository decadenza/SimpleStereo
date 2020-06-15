"""
active
======
Contains different active stereo algorithms and relative utilities.
"""
import os

import cv2


def generateGrayCodeImgs(targetDir, width, height, addHorizontal=False):
    """
    Generate Gray Codes and save it to PNG images.
    
    Starts from the couple of images *0.png* and *1.png* (one is the inverse of the other) containing vertical stripes.
    Then 2.png is coupled with 3.png and so on.
    The function stores also a *black.png* and *white.png* images for threshold calibration.
    The optional parameter addHorizontal (default is False) adds horizontal stripes too, appended after the vertical ones.
    
    Parameters
    ----------
    targetDir : string
        Path to the directory where to save Gray codes. Directory is created if not exists.
    width, height : int
        Pixel dimensions of the images.
    addHorizontal : bool, optional
        Whether append also horizontal patterns after the vertical ones.
    
    Returns
    -------
    int
        Number of generated patterns (black and white are not considered in this count).
    """
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
    cv2.imwrite( os.path.join(targetDir,'white.png'), (255*np.ones((height, width), np.uint8)) )  # white
    cv2.imwrite( os.path.join(targetDir,'black.png'), (np.zeros((height, width), np.uint8)) )     # black
    
    return num_patterns
