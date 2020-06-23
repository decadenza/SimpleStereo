"""
passive
=======
Contains different passive stereo algorithms to build disparity maps.

Simpler algorithms, like StereoBM and StereoSGBM, are already implemented in OpenCV.
"""
import ctypes

import numpy as np
import cv2

from simplestereo import passiveExt


class StereoASW():
    """
    Custom implementation of Adaptive Support Weight from "Locally adaptive support-weight approach
    for visual correspondence search", K. Yoon, I. Kweon, 2005.
    
    
    Parameters
    ----------
    winSize : int
        Side of the square window. Must be an odd positive number.
    maxDisparity: int
        Maximum accepted disparity. Default is 16.
    minDisparity: int
        Minimum valid disparity, usually set to zero. Default is 0.
    gammaC : int
        Color parameter. If increased, it increases the color influence. Default is 7.
    gammaP : int
        Proximity parameter. If increased, it increases the proximity influence. Default is 36.
    consistent : bool
        If True consistent check is made, i.e. disparity is calculated first using left image as reference,
        then using right one as reference. Any non-corresponding value is invalidated (occluded)
        and assigned as the nearest minimum left-right non-occluded disparity. Original idea from occlusion
        detection and filling as in "Local stereo matching using geodesic support weights", Asmaa Hosni et al., 2009.
        If enabled, running time is roughly doubled.
        Default to True.
        
    ..warning::
        It may get very slow for high resolution images or with high *winSize* or *maxDisparity* values.
    
    Notes
    -----
    - This algorithm performs a 384x288 pixel image scan with maxDisparity=16 in less than 1 sec
    using 4 CPUs (while other implementations need 60 sec, see DOI 10.1007/s11554-012-0313-2 with code "yk").
    - To improve the final result, a smoothering filter could be applied.

    """
    def __init__(self, winSize=11, maxDisparity=16, minDisparity=0, gammaC=7, gammaP=36, consistent=True): 
        
        if not (winSize>0 and winSize%2 == 1) :
            raise ValueError("winSize must be a positive odd number!")
            
        self.winSize = winSize
        self.maxDisparity = maxDisparity
        self.minDisparity = minDisparity
        self.gammaC = gammaC
        self.gammaP = gammaP
        self.consistent = consistent
        
    
    def compute(self, img1, img2):
        """
        Compute disparity map for BGR images.
        
        Parameters
        ----------
        img1, img2 : cv2.Mat
            A couple of OpenCV images (left and right, respectively) of same shape.
        
        Returns
        -------
        numpy.ndarray (np.int16)
            A disparity map of the same width and height of the images.
        """
        
        # Convert from BGR to CIELab
        img1Lab = cv2.cvtColor(img1.astype("float32") / 255, cv2.COLOR_BGR2Lab)
        img2Lab = cv2.cvtColor(img2.astype("float32") / 255, cv2.COLOR_BGR2Lab)
        
        # Send to C++ extension
        disparityMap = passiveExt.computeASW(img1, img2, img1Lab, img2Lab, self.winSize,
                                             self.maxDisparity, self.minDisparity,
                                             self.gammaC, self.gammaP, self.consistent)
        
        return disparityMap
        





################## TO BE COMPLETED...
class StereoGSW():
    """
    Tentative implementation of "Local stereo matching using geodesic support weights",
    Asmaa Hosni, Michael Bleyer, Margrit Gelautz and Christoph Rhemann (2009).
    
    This is only for educational purposes.
    The reference paper is not clear. Mutual Information computes a value for the whole
    window (not position based). However formula (5) suggests a per-pixel iteration.
    Currently implemented with sum of squared differences, weighted with geodesic.
    
    Parameters
    ----------
    winSize : int
        Side of the square window. Must be an odd positive number.
    maxDisparity: int
        Maximum accepted disparity. Default is 16.
    minDisparity: int
        Minimum valid disparity, usually set to zero. Default is 0.
    gamma : int
        Gamma parameter. If increased, it increases the geodesic weight influence. Default is 10.
    fMax : int or float
        Color difference is capped to this value. Default is 120.
    iterations : int
        Number of iteration for geodesic distances estimation. Default is 3.
    bins : int
        Number of bins for histograms (needed for Mutual Information). Default is 20.
        
    ..warning::
        This algorithm is very slow with high *winSize*. Do not use in production.
    
    ..todo::
        Right image reference and occlusion filling to be implemented. 
        Find a way to use Mutual information in matching cost.
    """
    def __init__(self, winSize=11, maxDisparity=16, minDisparity=0, gamma=10,
                 fMax=120, iterations=3, bins=20):
        
        if not (winSize>0 and winSize%2 == 1) :
            raise ValueError("winSize must be a positive odd number!")
            
        self.winSize = winSize
        self.gamma = gamma
        self.maxDisparity = maxDisparity
        self.minDisparity = minDisparity
        self.fMax = fMax
        self.iterations = iterations
        self.bins = bins
        
    def compute(self, img1, img2):
        """
        Compute disparity map for 3-color channel images.
        """
        
        # Send to C++ extension
        disparityMap = passiveExt.computeGSW(img1, img2, self.winSize,
                                             self.maxDisparity, self.minDisparity,
                                             self.gamma,self.fMax, self.iterations,
                                             self.bins)
        
        return disparityMap

