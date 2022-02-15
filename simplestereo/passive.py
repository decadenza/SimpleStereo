"""
passive
=======
Contains different passive stereo algorithms to build disparity maps.

Simpler algorithms, like StereoBM and StereoSGBM, are already implemented in OpenCV.
"""
#import ctypes

import numpy as np
import cv2

from simplestereo import _passive


class StereoASW():
    """
    Custom implementation of "Adaptive Support-Weight Approach
    for Correspondence Search", K. Yoon, I. Kweon, 2006.
    
    
    Parameters
    ----------
    winSize : int
        Side of the square window. Must be an odd positive number. Default is 35.
    maxDisparity: int
        Maximum accepted disparity. Default is 16.
    minDisparity: int
        Minimum valid disparity, usually set to zero. Default is 0.
    gammaC : float
        Color parameter. If increased, it increases the color influence. Default is 5.
    gammaP : float
        Proximity parameter. If increased, it increases the proximity influence. Default is 17.5.
    consistent : bool
        If True consistent check is made, i.e. disparity is calculated first using left image as reference,
        then using right one as reference. Any non-corresponding value is invalidated (occluded)
        and assigned as the nearest minimum left-right non-occluded disparity. Original idea from occlusion
        detection and filling as in "Local stereo matching using geodesic support weights", Asmaa Hosni et al., 2009.
        If enabled, running time is roughly doubled.
        Default to False.
    
    
    .. todo::
       Alternative version can be written like this: compute disparity map on every other pixel
       with the traditional algorithm, then fill the remaining pixels using left-right disparity
       boundaries. This proved to be 40-50% faster with no significant decrease in quality.
    
    
    .. warning::
       It gets very slow for high resolution images or with high *winSize* or *maxDisparity* values.
    
        
    .. note::
       This algorithm performs a 384x288 pixel image scan with maxDisparity=16 in less than 1 sec
       using 4 CPUs (while other implementations need 60 sec, see DOI 10.1007/s11554-012-0313-2 with code "yk").
       To improve the final result, a smoothering filter could be applied.

    """
    def __init__(self, winSize=35, maxDisparity=16, minDisparity=0, gammaC=5, gammaP=17.5, consistent=False): 
        
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
        
        # Send to C++ extension
        disparityMap = _passive.computeASW(img1, img2, self.winSize,
                                             self.maxDisparity, self.minDisparity,
                                             self.gammaC, self.gammaP, self.consistent)
                                             
        return disparityMap
        





class StereoGSW():
    """
    *Incomplete* implementation of "Local stereo matching using geodesic support weights",
    Asmaa Hosni, Michael Bleyer, Margrit Gelautz and Christoph Rhemann (2009).
    
    Parameters
    ----------
    winSize : int, optional
        Side of the square window. Must be an odd positive number.
    maxDisparity: int, optional
        Maximum accepted disparity. Default is 16.
    minDisparity: int, optional
        Minimum valid disparity, usually set to zero. Default is 0.
    gamma : int, optional
        Gamma parameter. If increased, it increases the geodesic weight influence. Default is 10.
    fMax : int or float, optional
        Color difference is capped to this value. Default is 120.
    iterations : int, optional
        Number of iteration for geodesic distances estimation. Default is 3.
    bins : int, optional
        Number of bins for histograms (currently not used, needed for Mutual Information). Default is 20.
        
    ..warning::
        Not optimized. Do not use in production.
    
    ..todo::
        This is a work in progress.
        The reference paper is not clear. Traditional Mutual Information computes a value for the whole
        window (not position based). However formula (5) suggests a per-pixel iteration.
        Currently implemented with sum of squared differences, weighted with geodesic.
        Need to implement Mutual information as matching cost.
        Need to implement right image consistency and subsequent occlusion filling. 
        
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
        disparityMap = _passive.computeGSW(img1, img2, self.winSize,
                                             self.maxDisparity, self.minDisparity,
                                             self.gamma,self.fMax, self.iterations,
                                             self.bins)
        
        return disparityMap

