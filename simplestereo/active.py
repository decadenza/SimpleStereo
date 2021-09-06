"""
active
======
Contains classes to manage active stereo algorithms and helper
functions.
This module contains both conventional active stereo (2 cameras +
projector) and structured-light (1 camera + projector) methods.
"""
import os

import numpy as np
import cv2
from scipy.interpolate import interp1d
import matplotlib                   # Temporary fix to avoid
matplotlib.use('TkAgg')             # segmentation fault error
import matplotlib.pyplot as plt

import simplestereo as ss

def generateGrayCodeImgs(targetDir, resolution):
    """
    Generate Gray Codes and save it to PNG images.
    
    Starts from the couple of images *0.png* and *1.png* (one is the inverse of the other).
    Then 2.png is coupled with 3.png and so on.
    First half contains vertical stripes, followed by horizontal ones.
    The function stores also a *black.png* and *white.png* images for
    threshold calibration.
    
    Parameters
    ----------
    targetDir : string
        Path to the directory where to save Gray codes. Directory is created if not exists.
    resolution : tuple
        Pixel dimensions of the images as (width, height) tuple (to be matched with projector resolution).
    
    Returns
    -------
    int
        Number of generated patterns (black and white are *not* considered in this count).
    """
    width, height = resolution
    graycode = cv2.structured_light_GrayCodePattern.create(width, height)
    
    num_patterns = graycode.getNumberOfPatternImages() # Surely an even number
    
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


def buildFringe(period=10, shift=0, dims=(1280,720), vertical=False, centralColor=None, dtype=np.uint8):
    """
    Build discrete sinusoidal fringe image.
    
    Parameters
    ----------
    period : float
        Fringe period along x axis, in pixels.
    shift : float
        Shift to apply. Default to 0.
    dims : tuple
        Image dimensions as (width, height).
    vertical : bool
        If True, fringe is build along vertical. Default to False (horizontal).
    centralColor : tuple
        BGR color for the central stripe. If None (default), no stripe is drawn and
        a grayscale image is returned.
    dtype: numpy.dtype
        Image is scaled in the range 0 - max value to match `dtype`.
        Default np.uint8 (max 255).
        
    Returns
    -------
    numpy.ndarray
        Fringe image.
    """
    
    if vertical is True:
        dims = (dims[1], dims[0]) # Swap dimensions
        
    row = np.iinfo(dtype).max * ((1 + np.cos(2*np.pi*(1/period)*(np.arange(dims[0], dtype=float) + shift)))/2)[np.newaxis,:]
    
    if centralColor is not None:
        row = np.repeat(row[:, :, np.newaxis], 3, axis=2)
        k = (dims[0]/2)//period
        left = int( period*(k - shift/(2*np.pi) - 0.5) )
        right = int(left+period)
        row[0, left:right, 0] *= centralColor[0]/255
        row[0, left:right, 1] *= centralColor[1]/255 
        row[0, left:right, 2] *= centralColor[2]/255 
        
    fullFringe = np.repeat(row.astype(dtype), dims[1], axis=0)
    
    if vertical is True:
        # Left->Right becomes Top->Bottom
        fullFringe = np.rot90(fullFringe, k=3, axes=(0,1))
        
    return fullFringe


    
def findCentralStripe(fringe, color, threshold=100):
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
        with np.errstate(divide='ignore',invalid='ignore'):
            # Some NaN expected.
            out = np.sum(img * i, axis=axis) / np.sum(img, axis=axis)
        return out
    
    x = getCenters(fringe,axis=1)
    y = np.arange(0.5,h,1)                # Consider pixel center, as first is in y=0.5
    mask = ~np.isnan(x)                   # Remove coords with NaN
    f = interp1d(y[mask],x[mask],kind="nearest",fill_value="extrapolate") # Interpolate
    x = f(y)
    
    return np.vstack((x, y)).T



########################################
###### (c) Pasquale Lafiosca 2020 ######
########################################
class StereoFTP:
    """
    Manager of the Stereo Fourier Transform Profilometry.
    
    Parameters
    ----------
    stereoRig : StereoRig
        A stereo rig object with camera in position 1 (world origin) and projector in
        position 2.
    fringeDims : tuple
        Dimensions of projector image as (width, height).
    period : float
        Period of the fringe (in pixels).
    stripeColor : tuple, optional
        BGR color used for the central stripe. Default to (0,0,255) (red).
    stripeThreshold : int, optional
        Threshold to find the stripe. See :func:`findCentralStripe()`.
    """
    
    def __init__(self, stereoRig, fringeDims, period,
                 stripeColor=(0,0,255), stripeThreshold=100):
        
        if fringe.ndim != 3:
            raise ValueError("fringe must be a BGR color image!")
        
        self.stereoRig = stereoRig
        self.fringeDims = fringeDims
        self.fp = 1/period
        self.stripeColor = stripeColor
        self.stripeThreshold = stripeThreshold
        
        '''
        # Initialization data
        self.projCoords, self.reference = self._getProjectorMapping()
        self.reference_gray = self.convertGrayscale(self.reference)
        self.Rectify1, self.Rectify2, self.Rotation = ss.rectification._lowLevelRectify(stereoRig)
        self.fc = self._calculateCameraFrequency()
        
        
        
        ### Get epipole on projector
        # Project camera position (0,0,0) onto projector image plane.
        ep = self.stereoRig.intrinsic2.dot(self.stereoRig.T)
        self.ep = ep/ep[2]
        '''
    
    @staticmethod
    def convertGrayscale(img):
        """
        Convert to grayscale using max.
        
        This keeps highest BGR value over the central stripe
        (e.g. (0,0,255) -> 255), allowing the FFT to work properly.
        
        Parameters
        ----------
        image : numpy.ndarray
            BGR image.
        
        Returns
        -------
        numpy.ndarray
            Grayscale image.
        """
        return np.max(img,axis=2)
    
    '''
    def _getProjectorMapping(self, interpolation = cv2.INTER_CUBIC):
        """
        Find projector image points corresponding to each camera pixel
        after projection on reference plane to build coordinates and
        virtual reference image (as seen from camera)
        
        Points are processed and returned in row-major order.
        The center of each pixel is considered as point.
        
        Returns
        -------
        Matrix of points with same width and height of camera resolution.
        
        Notes
        -----
        Corresponding points on reference plane do not vary. They have to
        be calculated only during initialization considering the chosen 
        reference plane.
        """
        
        w, h = self.stereoRig.res1
        invAc = np.linalg.inv(self.stereoRig.intrinsic1)
        
        # Build grid of x,y coordinates
        grid = np.mgrid[0:w,0:h].T.reshape(-1,1,2).astype(np.float64)
        # Consider center of pixel: it can be thought as
        # the center of the light beam entering the camera
        # Experiments showed that this is needed for projCoords
        # but *not* for the virtual reference image
        # (depends on how cv2.remap works, integer indexes
        # of original images are used)
        doubleGrid = np.vstack((grid+0.5, grid))
        doubleGrid = np.append(doubleGrid, np.ones((w*h*2,1,1), dtype=np.float64), axis=2)
        # For *both* grids
        # de-project from camera to reference plane
        # and project on projector's image plane.
        
        # 1st half: To get exact projector coordinates from camera x,y coordinates (center of pixel)
        # 2d half: To build a virtual reference image (using *integer* pixel coordinates)
        pp, _ = cv2.projectPoints(doubleGrid,
            self.lc*(self.stereoRig.R).dot(invAc), 
            self.stereoRig.T, self.stereoRig.intrinsic2,
            self.stereoRig.distCoeffs2)
        
        # Separate the two grids
        pointsA = pp[h*w:]                   # 1st half
        projCoords = pp[:h*w].reshape(h,w,2) # 2nd half
        
        mapx = ( pointsA[:,0,0] ).reshape(h,w).astype(np.float32)
        mapy = ( pointsA[:,0,1] ).reshape(h,w).astype(np.float32)
        
        virtualReferenceImg = cv2.remap(self.fringe, mapx, mapy, interpolation);
        
        return projCoords, virtualReferenceImg
    '''
    
    def _calculateCameraFrequency(self, z):
        """
        Estimate fc from system geometry, fp and z value.
        
        Draw a plane at given z distance in front of the camera.
        Find period size on it and project that size on camera.
        """
        Ac = self.stereoRig.intrinsic1
        Dc = self.stereoRig.distCoeffs1
        
        Ap = self.stereoRig.intrinsic2
        R = self.stereoRig.R
        T = self.stereoRig.T
        Dp = self.stereoRig.distCoeffs2
        
        Op = -np.linalg.inv(R).dot(T)
        ObjCenter = np.array([[[0],[0],[z]]], dtype=np.float32)
        
        # Send center of reference plane to projector
        pCenter, _ = cv2.projectPoints(ObjCenter, R, T, 
            self.stereoRig.intrinsic2, self.stereoRig.distCoeffs2)
        # Now we are in the projected image
        # Perfect fringe pattern. No distortion
        
        # Find two points at distance Tp (period on projector)
        halfPeriodP = (1/self.fp)/2
        
        leftP = pCenter[0,0,0] - halfPeriodP
        rightP = pCenter[0,0,0] + halfPeriodP
        points = np.array([[leftP,pCenter[0,0,1]], [rightP,pCenter[0,0,1]]])
        
        # Un-distort points for the projector means to distort
        # as the pinhole camera model is made for cameras
        # and we are projecting back to 3D world
        #print(points)
        distortedPoints = cv2.undistortPoints(points, Ap, Dp, P=Ap)
        
        # De-project points to reference plane
        invARp = np.linalg.inv(Ap.dot(R))
        M = np.array([[z-Op[2],0,Op[0]],[0,z-Op[2],Op[1]], [0,0,z]],dtype=np.object)
        
        wLeft = invARp.dot(np.append(distortedPoints[0,0,:], [1]))  # Left point
        wRight = invARp.dot(np.append(distortedPoints[1,0,:], [1])) # Right point
        
        # Normalize
        wLeft = wLeft/wLeft[2]
        wRight = wRight/wRight[2]
        
        # De-project
        wLeft = M.dot(wLeft)
        wRight = M.dot(wRight)
        
        # Put in shape (2,1,3)
        wPoints = np.array([ [wLeft],[wRight] ],dtype=np.float32)
        
        # Project on camera image plane (also applying lens distortion).
        # b points are like seen by the camera!
        b, _ = cv2.projectPoints(wPoints, np.eye(3), np.zeros((3,1)), Ac, Dc)
        
        # Now we have 2 points on the camera that differ
        # exactly one projector period from each other
        # as seen by the camera
        # Use the first Euclid theorem to get the wanted period
        Tc = ((b[0,0,0] - b[1,0,0])**2 + (b[0,0,1] - b[1,0,1])**2)/abs(b[0,0,0]-b[1,0,0])
        
        # Return frequency
        return 1/Tc    
    
    def _triangulate(self, c_x, c_y, p_x):
        """
        Given camera point (c_x, c_y) and projector x value p_x
        find 3D point.
        """
        pass
    
    def getCloud(self, imgObj, fc=None, radius_factor=0.5, roi=None, unwrappingMethod=None, plot=False):
        """
        Process an image and get the point cloud.
        
        Parameters
        ----------
        imgObj : numpy.ndarray
            BGR image acquired by the camera.
        fc : float, optional
            Fundamental frequency from the camera. If None,
            it is calculated automatically. Default to None.
        radius_factor : float, optional
            Radius factor of the pass-band filter. Default to 0.5.
        roi : tuple, optional
            Region of interest in the format (x,y,width,height)
            where x,y is the upper left corner. Default to None.
        unwrappingMethod : function, optional
            Pointer to chosen unwrapping function. It must take the phase
            as the only argument. If None (default), `np.unwrap`is used.
            
        Returns
        -------
        Point cloud with shape (height,width,3), with height and width 
        same as the input image or selected `roi`.
        """
        
        # Check that is a color image
        if imgObj.ndim != 3:
            raise ValueError("image must be a BGR color image!")
        
        
        widthC, heightC = self.stereoRig.res1 # Camera resolution
        
        if roi is not None:
            # Cut image to given roi
            roi_x, roi_y, roi_w, roi_h = roi
            imgObj = imgObj[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
        else:
            roi_x,roi_y,roi_w,roi_h = (0,0,widthC,heightC)
            
        
        if fc is None:
            # If filtering frequency is not given, estimate from red line.
            
            # Find central stripe on camera image
            cs = ss.active.findCentralStripe(imgObj,
                                    self.stripeColor, self.stripeThreshold)
            cs = cs.reshape(-1,1,2).astype(np.float64)
            cs = cv2.undistortPoints(cs,               # Consider distortion
                            self.stereoRig.intrinsic2,
                            self.stereoRig.distCoeffs2,
                            P=self.stereoRig.intrinsic2)
            stripe = cs.reshape(-1,2) # x, y camera points
            
            ### Find world points corresponding to stripe
            
            
            
        
        
        
        
        
        
        '''
            
        widthC, heightC = self.stereoRig.res1 # Camera resolution
        imgR = self.reference
        imgR_gray = self.reference_gray
        projCoords = self.projCoords
        imgObj = cv2.undistort(image, self.stereoRig.intrinsic1, self.stereoRig.distCoeffs1)
        
        if roi is not None:
            roi_x,roi_y,roi_w,roi_h = roi
            # Cut images to roi
            imgObj = imgObj[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
            imgR = imgR[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
            imgR_gray = imgR_gray[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
            projCoords = projCoords[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
        else:
            roi_x,roi_y,roi_w,roi_h = (0,0,widthC,heightC)
            
        # Preprocess image for phase analysis
        imgObj_gray = self.convertGrayscale(imgObj)
        
        ### Phase analysis
        G0 = np.fft.fft(imgR_gray, axis=1)     # FFT on x dimension
        G = np.fft.fft(imgObj_gray, axis=1)
        freqs = np.fft.fftfreq(roi_w)
        
        # Pass-band filter
        radius = radius_factor*fc
        fmin = fc - radius
        fmax = fc + radius
        
        fIndex = min(range(len(freqs)), key=lambda i: abs(freqs[i]-fc))
        fminIndex = min(range(len(freqs)), key=lambda i: abs(freqs[i]-fmin))
        fmaxIndex = min(range(len(freqs)), key=lambda i: abs(freqs[i]-fmax))
        
        if plot:
            cv2.namedWindow('Virtual reference',cv2.WINDOW_NORMAL)
            cv2.namedWindow('Object',cv2.WINDOW_NORMAL)
            cv2.imshow("Virtual reference", imgR)
            cv2.imshow("Object", imgObj)
            print("Press a key over the images to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            plt.suptitle("Middle row absolute phase")
            # Show module of FFTs
            plt.plot(freqs[:roi_w//2], np.absolute(G0[roi_h//2-1,:roi_w//2]), label="|G0|", linestyle='--', color='red')
            plt.plot(freqs[:roi_w//2], np.absolute(G[roi_h//2-1,:roi_w//2]), label="|G|", linestyle='-', color='blue')
            # Show filtered band
            plt.axvline(x=freqs[fIndex], linestyle='-', color='black')
            plt.axvline(x=freqs[fminIndex], linestyle='dotted', color='black')
            plt.axvline(x=freqs[fmaxIndex], linestyle='dotted', color='black')
            
            plt.title(f"fc={fc}", size="small")    
            plt.legend()
            plt.show()
            plt.close()
        
        ### Phase filtering
        # The band-pass filter introduces some systematic error.
        # Larger the filter, more complex shape are followed, but
        # more noise is introduced (aliasing).
        
        # Reference image filtering
        G0[:,:fminIndex] = 0
        G0[:,fmaxIndex+1:] = 0
        g0hat = np.fft.ifft(G0,axis=1)
        
        # Object image filtering
        G[:,:fminIndex] = 0
        G[:,fmaxIndex+1:] = 0
        ghat = np.fft.ifft(G,axis=1)
            
        
        # Build the new signal and get its phase
        newSignal = ghat * np.conjugate(g0hat)
        phase = np.angle(newSignal)
        
        if unwrappingMethod is None:
            # Unwrap along the direction perpendicular to the fringe
            phaseUnwrapped = np.unwrap(phase, axis=1)
        else:
            phaseUnwrapped = unwrappingMethod(phase)
              
        if plot:
            plt.title("Middle row phase")
            plt.plot(np.arange(roi_w), phase[roi_h//2-1,:], label="Phase", linestyle='-.', color='red')
            plt.plot(np.arange(roi_w), phaseUnwrapped[roi_h//2-1,:], label="Unwrapped phase", linestyle='-', color='blue')
            plt.xlabel("Pixel position", fontsize=20)
            plt.ylabel("Phase", fontsize=20)
            plt.legend(fontsize=12)
            plt.show()
            plt.close()
        
        ### Lazy shortcut for many values
        Ac = self.stereoRig.intrinsic1
        Dc = self.stereoRig.distCoeffs1
        
        Ap = self.stereoRig.intrinsic2
        R = self.stereoRig.R
        T = self.stereoRig.T
        Dp = self.stereoRig.distCoeffs2
        ep = self.ep
        
        ### Find k values from central stripe
        # Triangulation can be done directly between left and right red stripe
        # and then compare with the phase value, but, as a quicker method
        # we find approximated A and H points over projector image plane
        # and count the integer periods of shift between the two
        
        # Find central line on the undistorted object image
        objStripe = ss.active.findCentralStripe(imgObj, self.stripeColor,
                        self.stripeThreshold)
        
        # Find integer indexes (round half down)
        # Accept loss of precision as k values must be rounded to integers
        cam_indexes = np.ceil(objStripe-0.5).astype(np.int) # As (x,y)
        
        pointA = projCoords[cam_indexes[:,1],cam_indexes[:,0]] # As (x,y)
        
        # Find Hx and Ax coords and calculate vector k
        pointA_y_rounded = np.ceil(pointA[:,1]-0.5).astype(np.int) # Round half down
        pointH_x = self.fringeStripe[pointA_y_rounded,0] # Get all x stripe coords
        theta = phaseUnwrapped[cam_indexes[:,1],cam_indexes[:,0]]
        k = (pointH_x - pointA[:,0])*self.fp - theta/(2*np.pi)
        k = np.rint(k).reshape(-1,1) # Banker's rounding + appropriate reshape
        
        
        # Adjust phase using k values
        phaseUnwrapped = phaseUnwrapped + k * 2*np.pi
        phaseUnwrapped = phaseUnwrapped.reshape(-1,1)
        
        # Get A and B points in pixels on imgFringe
        Xa = projCoords[:,:,0].reshape(-1,1)
        Ya = projCoords[:,:,1].reshape(-1,1)
        
        Xh = Xa + phaseUnwrapped/(2*np.pi*self.fp)
        # Find y coord on epipolar line
        Yh = ( (Xh-ep[0])/(Xa-ep[0]) )*(Ya-ep[1]) + ep[1]
            
        # Desidered point is H(Xh,Yh)
        H = np.hstack((Xh,Yh)).reshape(-1,1,2).astype(np.float64)
        
        # *Apply* lens distortion to H.
        # A projector is considered as an inversed pinhole camera (and so are
        # the distortion coefficients)
        # H is on the original imgFringe. Passing through the projector lenses,
        # it gets distortion, so it does not coincide with real world point.
        # But we want rays going exactly towards world points.
        # Remove intrinsic, undistort and put same intrinsic back.
        H = cv2.undistortPoints(H, Ap, Dp, P=Ap)
        
        ### Triangulation
        
        ### Get inverse common orientation and extend to 4x4 transform
        R_inv = np.linalg.inv(self.Rotation)
        R_inv = np.hstack( ( np.vstack( (R_inv,np.zeros((1,3))) ), np.zeros((4,1)) ) )
        R_inv[3,3] = 1
        
        # Build grid of indexes and apply rectification (undistorted camera points)
        pc = np.mgrid[0:widthC,0:heightC].T
        pc = pc[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w].reshape(-1,1,2).astype(np.float64)
        # Consider pixel center (see also projCoords in self._getProjectorMapping)
        pc = pc + 0.5
        pc = cv2.perspectiveTransform(pc, self.Rectify1).reshape(-1,2) # Apply rectification
        # Add ones as third coordinate
        pc = np.hstack( (pc,np.ones((roi_w*roi_h,1),dtype=np.float64)) )
        
        # Apply rectification to projector points.
        # Rectify2 cancels the intrinsic and applies new rotation.
        # No new intrinsics here.
        pp = cv2.perspectiveTransform(H, self.Rectify2).reshape(-1,2)
        
        # Get world points
        disparity = np.abs(pp[:,[0]] - pc[:,[0]])
        pw = self.stereoRig.getBaseline()*(pc/disparity)
        
        # Cancel common orientation applied to first camera
        # to bring points into camera coordinate system
        finalPoints = cv2.perspectiveTransform(pw.reshape(-1,1,3), R_inv)
        
        # Reshape as original image    
        return finalPoints.reshape(roi_h,roi_w,3)

        '''

class GrayCode:
    """
    Wrapper for the Gray code method from OpenCV.
    
    Structured-light implementation using camera-projector
    calibrated rig.
    
    Parameters
    ----------
    rig : StereoRig
        A stereo rig object with camera in position 1 (world origin) and projector in
        position 2.
    black_thr : int, optional
       Black threshold is a number between 0-255 that represents the
       minimum brightness difference required for valid pixels, between
       the fully illuminated (white) and the not illuminated images
       (black); used in computeShadowMasks method. Default to 40.
    white_thr : int, optional
        White threshold is a number between 0-255 that represents the
        minimum brightness difference required for valid pixels, between
        the graycode pattern and its inverse images; used in 
        `getProjPixel` method. Default to 5.
    
    Notes
    -----
    Projector distortion may be unaccurate, especially along border.
    If this is the case, you can ignore it setting
    `rig.distCoeffs2 == None` before passing `rig` to the constructor
    or setting a narrow `roi`.
    """
    def __init__(self, rig, black_thr=40, white_thr=5):
        self.rig = rig
        # Build graycode using projector resolution
        self.graycode = cv2.structured_light_GrayCodePattern.create(rig.res2[0], rig.res2[1])
        self.graycode.setBlackThreshold(black_thr)
        self.graycode.setWhiteThreshold(white_thr)
        self.num_patterns = self.graycode.getNumberOfPatternImages()
        self.Rectify1, self.Rectify2, commonRotation = ss.rectification._lowLevelRectify(rig)
        # Get inverse common orientation and extend to 4x4 transform
        self.R_inv = np.linalg.inv(commonRotation)
        self.R_inv = np.hstack( ( np.vstack( (self.R_inv,np.zeros((1,3))) ), np.zeros((4,1)) ) )
        self.R_inv[3,3] = 1
        
    def getCloud(self, images, roi=None):
        """
        Perform the 3D point calculation from a list of images.
        
        Parameters
        ----------
        images : list or tuple       
            A list of image *paths* acquired by the camera.
            Each set must be ordered like all the Gray code patterns
            (see `cv2.structured_light_GrayCodePattern`).
            Any following extra image will be ignored (es. full white).
        roi : tuple, optional
            Region of interest in the format (x,y,width,height)
            where x,y is the upper left corner. Default to None.
        
        Returns
        -------
        numpy.ndarray
            Points with shape (n,1,3)
        
        ..todo::
            Add possibility to return point cloud in same image/roi
            shape with NaN in invalid locations.
        """
        widthC, heightC = self.rig.res1 # Camera resolution
        imgs = []
        
        # Load images
        for fname in images[:self.num_patterns]: # Exclude any extra images (es. white, black)
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img.shape != (heightC,widthC):
                raise ValueError(f'Image size of {fname} is mismatch!')
            img = cv2.undistort(img, self.rig.intrinsic1, self.rig.distCoeffs1)
            imgs.append(img)
        
        pc = []
        pp = []
        
        if roi is not None:
            roi_x,roi_y,roi_w,roi_h = roi
        else:
            roi_x,roi_y,roi_w,roi_h = (0, 0, widthC, heightC)
            
        # Find corresponding points
        # Since we are jumping some points, there is no correspondence
        # with any BGR image used for color in the final PLY file.
        for y in range(roi_y,roi_h):
            for x in range(roi_x,roi_w):
                err, proj_pix = self.graycode.getProjPixel(imgs, x, y)
                if not err:
                    pp.append(proj_pix)
                    pc.append((x,y))
        
        # Now we have solved the stereo correspondence problem.
        # To do triangulation easily, it is better to rectify.
        
        # Convert
        pc = np.asarray(pc).astype(np.float64).reshape(-1,1,2)
        pp = np.asarray(pp).astype(np.float64).reshape(-1,1,2)
        
        # Consider pixel center (negligible difference, anyway...)
        pc = pc + 0.5
        pp = pp + 0.5
        
        # *Apply* lens distortion to pp.
        # A projector is considered as an inversed pinhole camera (and so are
        # the distortion coefficients)
        # H is on the original imgFringe. Passing through the projector lenses,
        # it gets distortion, so it does not coincide with real world point.
        # But we want points to be an exact projection of the world points.
        # Remove intrinsic, undistort and put same intrinsic back.
        pp = cv2.undistortPoints(pp, self.rig.intrinsic2, self.rig.distCoeffs2, P=self.rig.intrinsic2)
        
        # Apply rectification
        pc = cv2.perspectiveTransform(pc, self.Rectify1).reshape(-1,2)
        pp = cv2.perspectiveTransform(pp, self.Rectify2).reshape(-1,2)
        
        # Add ones as third coordinate
        pc = np.hstack( (pc,np.ones((pc.shape[0],1),dtype=np.float64)) )
        
        # Get world points
        disparity = np.abs(pp[:,[0]] - pc[:,[0]])
        pw = self.rig.getBaseline()*(pc/disparity)
        
        # Cancel common orientation applied to first camera
        # to bring points into camera coordinate system
        finalPoints = cv2.perspectiveTransform(pw.reshape(-1,1,3), self.R_inv)
        
        return finalPoints
        

# Alias for single camera version
GrayCodeSingle = GrayCode

class GrayCodeDouble:
    """
    Wrapper for the Gray code method from OpenCV.
    
    Conventional active stereo implementation, i.e. using two calibrated cameras and
    one uncalibrated projector.
    
    Parameters
    ----------
    rig : StereoRig
        A stereo rig object with two cameras.
    projRes : tuple
        Projector resolution as (width, height).
    black_thr : int, optional
       Black threshold is a number between 0-255 that represents the
       minimum brightness difference required for valid pixels, between
       the fully illuminated (white) and the not illuminated images
       (black); used in computeShadowMasks method. Default to 40.
    white_thr : int, optional
        White threshold is a number between 0-255 that represents the
        minimum brightness difference required for valid pixels, between
        the graycode pattern and its inverse images; used in 
        `getProjPixel` method. Default to 5.
    
    Notes
    -----
    Projector distortion may be unaccurate, especially along border.
    If this is the case, you can ignore it setting
    `rig.distCoeffs2 == None` before passing `rig` to the constructor
    or setting a narrow `roi`.
    """
    def __init__(self, rig, projRes, black_thr=40, white_thr=5):
        self.rig = rig
        self.projRes = projRes
        # Build graycode using projector resolution
        self.graycode = cv2.structured_light_GrayCodePattern.create(projRes[0], projRes[1])
        self.graycode.setBlackThreshold(black_thr)
        self.graycode.setWhiteThreshold(white_thr)
        self.num_patterns = self.graycode.getNumberOfPatternImages()
        self.Rectify1, self.Rectify2, commonRotation = ss.rectification._lowLevelRectify(rig)
        ### Get inverse common orientation and extend to 4x4 transform
        self.R_inv = np.linalg.inv(commonRotation)
        self.R_inv = np.hstack( ( np.vstack( (self.R_inv,np.zeros((1,3))) ), np.zeros((4,1)) ) )
        self.R_inv[3,3] = 1
        
    def getCloud(self, images, roi1=None, roi2=None):
        """
        Perform the 3D point calculation from a list of images.
        
        Parameters
        ----------
        images : list or tuple       
            A list (or tuple) of 2 dimensional tuples (ordered left and
            right) of image paths, e.g. [("oneL.png","oneR.png"),
            ("twoL.png","twoR.png"), ...]
            Each set must be ordered like all the Gray code patterns
            (see `cv2.structured_light_GrayCodePattern`).
            Any following extra image will be ignored (es. full white).
        roi1 : tuple, optional
            Region of interest on camera 1 in the format
            (x,y,width,height) where x,y is the upper left corner.
            Default to None.
        roi2 : tuple, optional
            As `roi1`, but for camera 2.
        
        Returns
        -------
        numpy.ndarray
            Points with shape (n,1,3)
        """
        w1, h1 = self.rig.res1 # Camera 1 resolution
        w2, h2 = self.rig.res2 # Camera 1 resolution
        # Contains at proj indexes, both camera correspondences as (x1,y1,x2,y2)
        corresp = np.full((self.projRes[1], self.projRes[0], 4), -1, dtype=np.intp)
        
        # Load images
        imgs1 = []
        imgs2 = []
        for fname1, fname2 in images[:self.num_patterns]: # Exclude any extra images (es. white, black)
            img1 = cv2.imread(fname1, cv2.IMREAD_GRAYSCALE)
            if img1.shape != (h1,w1):
                raise ValueError(f'Image size of {fname1} is mismatch!')
            img2 = cv2.imread(fname2, cv2.IMREAD_GRAYSCALE)
            if img2.shape != (h2,w2):
                raise ValueError(f'Image size of {fname2} is mismatch!')
            img1 = cv2.undistort(img1, self.rig.intrinsic1, self.rig.distCoeffs1)
            img2 = cv2.undistort(img2, self.rig.intrinsic2, self.rig.distCoeffs2)
            imgs1.append(img1)
            imgs2.append(img2)
        
        if roi1 is not None:
            roi1_x,roi1_y,roi1_w,roi1_h = roi1
        else:
            roi1_x,roi1_y,roi1_w,roi1_h = (0, 0, w1, h1)
            
        # Find corresponding points
        for y in range(roi1_y,roi1_h):
            for x in range(roi1_x,roi1_w):
                err, proj_pix = self.graycode.getProjPixel(imgs1, x, y)
                if not err:
                    corresp[y,x,0] = proj_pix[0]
                    corresp[y,x,1] = proj_pix[1]
        
        if roi2 is not None:
            roi2_x,roi2_y,roi2_w,roi2_h = roi2
        else:
            roi2_x,roi2_y,roi2_w,roi2_h = (0, 0, w2, h2)
            
        # Find corresponding points
        for y in range(roi2_y,roi2_h):
            for x in range(roi2_x,roi2_w):
                err, proj_pix = self.graycode.getProjPixel(imgs2, x, y)
                if not err:
                    corresp[y,x,2] = proj_pix[0]
                    corresp[y,x,3] = proj_pix[1]
        
        # Filter out missing correspondences
        corresp = corresp[(corresp>-1).any(axis=2)]
        
        # Consider pixel center (negligible difference, anyway...)
        corresp += 0.5
        
        
        # Now we have solved the stereo correspondence problem.
        # To do triangulation easily, it is better to rectify.
        
        # Convert
        p1 = corresp[:,:,:2].astype(np.float64).reshape(-1,1,2)
        p2 = corresp[:,:,2:4].astype(np.float64).reshape(-1,1,2)
        
        # Apply rectification
        p1 = cv2.perspectiveTransform(p1, self.Rectify1).reshape(-1,2)
        p2 = cv2.perspectiveTransform(p2, self.Rectify2).reshape(-1,2)
        
        # Add ones as third coordinate
        p1 = np.hstack( (p1,np.ones((pc.shape[0],1),dtype=np.float64)) )
        
        # Get world points
        disparity = np.abs(p1[:,[0]] - p2[:,[0]])
        pw = self.rig.getBaseline()*(p1/disparity)
        
        # Cancel common orientation applied to first camera
        # to bring points into camera coordinate system
        finalPoints = cv2.perspectiveTransform(pw.reshape(-1,1,3), self.R_inv)
        
        return finalPoints


def computeROI(img, blackThreshold=10, extraMargin=0):
    """
    Exclude black regions along borders.
    
    Usually the projector does not illuminate the whole image area.
    This function attempts to find the region of interest as a rectangle
    inside the biggest contour.
    
    Parameters
    ----------
    img : numpy.ndarray
        Camera image with black borders.
    blackThreshold : int, optional
        Threshold for the black regions between 0 and 255.
        Default to 10.
    extraMargin : int, optional
        Extra safety margin to reduce to ROI. Default to 0.
    Returns
    -------
    tuple
        ROI as tuple of integers (x,y,w,h).
     
    """
    if img.ndim>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height,width = img.shape
    _, img_thresh = cv2.threshold(img, blackThreshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Select biggest contour
    best_cnt = max(contours, key = cv2.contourArea)
    # Find bounding rectangle
    x,y,w,h = cv2.boundingRect(best_cnt)
    
    # Look around rect borders and start shinking until is all inside
    x2,y2,w2,h2 = x,y,w,h
    while(True):
        allInside = True
        # TOP
        for j in range(x2,x2+w2):
            if cv2.pointPolygonTest(best_cnt, (j,y2), False)<0: # Point is outside
                y2+=1
                allInside = False
                break
        # RIGHT
        for i in range(y2,y2+h2):
            if cv2.pointPolygonTest(best_cnt, (x2+w2,i), False)<0:
                w2-=1
                allInside = False
                break
        # BOTTOM
        for j in range(x2,x2+w2):
            if cv2.pointPolygonTest(best_cnt, (j,y2+h2), False)<0:
                h2-=1
                allInside = False
                break
        # LEFT
        for i in range(y2,y2+h2):
            if cv2.pointPolygonTest(best_cnt, (x2,i), False)<0:
                x2+=1
                allInside = False
                break
        
        if allInside: # all the rect borders are within the contour
            break
    
    # Reduce ROI further.
    x2 += extraMargin
    y2 += extraMargin
    w2 -= extraMargin
    h2 -= extraMargin
        
    return (x2,y2,w2,h2)
