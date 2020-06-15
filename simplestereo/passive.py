"""
passive
=======
Contains different passive stereo algorithms to build disparity maps.

See also algorithms implemented in OpenCV:
    - StereoBM
    - StereoSGBM
"""
import ctypes

import numpy as np
import cv2

from simplestereo import passiveExt


class StereoASW():
    
    def __init__(self, winSize=11, maxDisparity=16, minDisparity=0, gammaC=7, gammaP=36): 
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
            Color parameter. Default is 7.
        gammaP : int
            Proximity parameter. Default is 36.
            
        ..warning::
            Not optimized. Very slow for high resolution images or with high *winSize* or *maxDisparity* values.
        
        Notes
        -----
        This algorithm performs a 384x288 pixel image scan in about 60 sec (see DOI 10.1007/s11554-012-0313-2 with code "yk").
        Optimized version is needed.

        """
        if not (winSize>0 and winSize%2 == 1) :
            raise ValueError("winSize must be a positive odd number!")
            
        self.winSize = winSize
        self.maxDisparity = maxDisparity
        self.minDisparity = minDisparity
        self.gammaC = gammaC
        self.gammaP = gammaP
        
    
    def compute(self, img1, img2):
        """
        Compute disparity map for BGR images.
        
        Parameters
        ----------
        img1, img2 : cv2.Mat
            A couple of OpenCV images (left and right, respectively) of same shape.
        
        Returns
        -------
        numpy.ndarray
            A disparity map of the same width and height of the images.
        """
        
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimension!")
        
        # Convert from BGR to CIELab
        img1Lab = cv2.cvtColor(img1.astype("float32") / 255, cv2.COLOR_BGR2Lab)
        img2Lab = cv2.cvtColor(img2.astype("float32") / 255, cv2.COLOR_BGR2Lab)
        
        # Send to C++ extension
        disparityMap = passiveExt.computeASW(img1, img2, img1Lab, img2Lab, self.winSize,
                                             self.maxDisparity, self.minDisparity,
                                             self.gammaC, self.gammaP)
        
        return disparityMap
        




################## TO BE COMPLETED...
class StereoGSW():
    
    def __init__(self, winSize=11, maxDisparity=16, minDisparity=0, gamma=10, fMax=120, iterations=3):
        """
        Implementation of "Local stereo matching using geodesic support weights",
        Asmaa Hosni, Michael Bleyer, Margrit Gelautz and Christoph Rhemann (2009).
        
        ..warning::
            VERY SLOW! NOT READY FOR PRODUCTION!
        """
        if not (winSize>0 and winSize%2 == 1) :
            raise ValueError("winSize must be a positive odd number!")
            
        self.winSize = winSize
        self.gamma = gamma
        self.fMax = fMax
        self.iterations = iterations
        self.maxDisparity = maxDisparity
        self.minDisparity = minDisparity
    
    def compute(self, img1, img2):
        """
        Compute disparity map for 3-color channel images.
        """
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimension!")
        
        height, width = img1.shape[:2]
        winSize = self.winSize
        padding = (winSize-1)//2
        
        # Add a zero border around images to simplify iteration, convert to float and rescale in 0-1
        img1 = cv2.copyMakeBorder(img1, top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_REPLICATE).astype(np.float32) / 255
        img2 = cv2.copyMakeBorder(img2, top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_REPLICATE).astype(np.float32) / 255
        
        # Initialize map
        disparityMap = np.zeros( (height, width), dtype=np.int16)
        
        # Strides (NOT REALLY SPEEDING UP)
        newShape = (img1.shape[0]-padding, img1.shape[1]-padding, winSize, winSize, 3)
        newStrides = img1.strides[:2] * 2 + (img1.strides[2],)
        img1Win = np.lib.stride_tricks.as_strided(img1, shape=newShape, strides=newStrides, writeable=False)
        img2Win = np.lib.stride_tricks.as_strided(img2, shape=newShape, strides=newStrides, writeable=False)
        
        # Working variables
        c = (winSize*winSize-1)//2
        tot = winSize*winSize
        
        def getGeodesicMap(win):
            # Compute weights of support window having its *upper left* corner in y,x
            cost = np.full(tot, np.inf, dtype=np.float32)
            cost[c] = 0
            win = win.reshape(tot,3)
            
            # Number of iterations
            for _ in range(self.iterations):
                # Forward
                for i in range(tot):
                    cost[i] = np.min( cost[:c+1] + np.linalg.norm(win[i] - win[c]) )
                # Backward
                for i in reversed(range(tot)):
                    cost[i] = np.min( cost[c:] + np.linalg.norm(win[i] - win[c]) )    
            
            cost = np.exp(-cost/self.gamma)    
            
            return cost.reshape(winSize,winSize)
        
        
        # Compute disparity using left image as reference
        for y in range(height):
            for x in range(width):
                color1 = img1[y:y+winSize,x:x+winSize]
                w1 = getGeodesicMap(color1)
                dBest = 0
                costBest = np.inf
                #for d in range(x+1): # Add -minDisparity here
                for d in reversed(range(max(x-self.maxDisparity,0), x+1-self.minDisparity)):
                    # Compute color differences between left and right
                    crossDiff = np.linalg.norm(color1 - img2[y:y+winSize,d:d+winSize], axis=2)
                    crossDiff[crossDiff>self.fMax] = self.fMax
                    cost = np.sum(w1 * crossDiff)
                    if cost < costBest:
                        costBest = cost
                        dBest = x-d
                
                disparityMap[y,x] = dBest
            
            print(y)
        
        # Check consistency using right image as reference
        for y in range(height):
            for x in range(width):
                color2 = img2[y:y+winSize,x:x+winSize]
                w2 = getGeodesicMap(color2)
                dBest = 0
                costBest = np.inf
                #for d in range(x,width): # Add -minDisparity here
                for d in range(x+self.minDisparity, min(x+self.maxDisparity+1,width)):
                    crossDiff = np.linalg.norm(color2 - img1[y:y+winSize,d:d+winSize], axis=2)
                    crossDiff[crossDiff>self.fMax] = self.fMax
                    cost = np.sum(w2 * crossDiff)
                    if cost < costBest:
                        costBest = cost
                        dBest = d-x
                        
                if disparityMap[y,x+dBest] != dBest:
                    disparityMap[y,x+dBest] = -1
                
            print(y, "bis")
            
            
            
        # PIXEL INVALIDATION AND SMOOTHING
        for y in range(height):
            for x in range(width):
                if disparityMap[y,x] == -1:
                    left = x-1
                    while(left>=0 and disparityMap[y,left] == -1):
                        left-=1
                    
                    right = x+1
                    while(right<width and disparityMap[y,right] == -1):
                        right+=1
                    
                    # Ensure that we are within image limits
                    if left < 0:
                        disparityMap[y,0:right] = disparityMap[y,right]
                    elif right > width-1:
                        disparityMap[y,left:width] = disparityMap[y,left]
                    else:
                        disparityMap[y,left:right] = min(disparityMap[y,left], disparityMap[y,right]) 
        
        
        # Da fare: applicare smoothering filter usando conservando i pesi calcolati prima (?)
        
                   
        return disparityMap #.astype(np.uint8)

