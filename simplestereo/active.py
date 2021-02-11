"""
active
======
Contains different active stereo algorithms and relative utilities.
"""
import os

import numpy as np
import cv2
from scipy.interpolate import interp1d
import matplotlib                   # Temporary fix to avoid
matplotlib.use('TkAgg')             # segmentation fault error
import matplotlib.pyplot as plt

import simplestereo as ss

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
        mask = ~np.isnan(y)                   # Remove coords with NaN
        f = interp1d(x[mask],y[mask],kind="nearest",fill_value="extrapolate") # Interpolate
        y = f(x)
        
    else:
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
    
    stereoRig : simplestereo.StereoRig object
        A stereo rig object with camera in position 1 (world origin) and projector in
        position 2.
    lc : float
        Approximate distance from the reference plane (in world units). This does not affect 3D points
        but the reference image. You should ensure that no black border appears on your
        region of interest (ROI).
    fringe : numpy.ndarray
        Image of the original projected fringe. 
    period : float
        Period of the fringe (in pixels).
    horizontal : bool
        Fringe orientation. Default to False (vertical stripes).
    """
    
    def __init__(self, stereoRig, lc, fringe, period, horizontal=False,
                 stripeColor=(0,0,255), stripeThreshold=100):
        
        if fringe.ndim != 3:
            raise ValueError("fringe must be a BGR color image!")
        
        self.stereoRig = stereoRig
        self.fringe = fringe
        self.fp = 1/period
        self.horizontal = horizontal
        self.lc = lc
        self.stripeColor = stripeColor
        self.stripeThreshold = stripeThreshold
        
        # Initialization data
        self.projCoords, self.reference = self._getProjectorMapping()
        self.reference_gray = self.convertGrayscale(self.reference)
        self.Rectify1, self.Rectify2, self.Rotation = self._getRectification()
        self.fc = self._calculateCameraFrequency()
        
        ### Find central stripe on fringe image
        cs = ss.active.findCentralStripe(self.fringe,
                                stripeColor, stripeThreshold, horizontal)
        cs = cs.reshape(-1,1,2).astype(np.float64)
        cs = cv2.undistortPoints(cs,               # Consider distortion
                        self.stereoRig.intrinsic2,
                        self.stereoRig.distCoeffs2,
                        P=self.stereoRig.intrinsic2)
        # One pixel spaced (x,y) coords of the stripe
        # over the projector image height (width if horizontal is True)
        self.fringeStripe = cs.reshape(-1,2)
        
        ### Get epipole on projector
        # Project camera position (0,0,0) onto projector image plane.
        ep = self.stereoRig.intrinsic2.dot(self.stereoRig.T)
        self.ep = ep/ep[2]
    
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
    
    
    def _getRectification(self):
        """
        Get basic rectification using Fusiello et al.
        for *internal* purposes only.
        
        This assumes that camera is in world origin.
        Please refer to the rectification module for general
        image rectification.
        
        See Also
        --------
        :meth:`simplestereo.rectification.fusielloRectify`
        """
        
        # Get baseline vector
        _, B = self.stereoRig.getCenters()
        # Find new directions
        v1 = B                          # New x direction
        v2 = np.cross([0,0,1], v1)      # New y direction
        v3 = np.cross(v1,v2)            # New z direction
        # Normalize
        v1 = v1 / np.linalg.norm(v1)    # Normalize x
        v2 = v2 / np.linalg.norm(v2)    # Normalize y
        v3 = v3 / np.linalg.norm(v3)    # Normalize z
        # Create rotation matrix
        R = np.array( [ v1, v2, v3 ] )
        
        # Build rectification transforms
        R1 = ( R ).dot( np.linalg.inv(self.stereoRig.intrinsic1) )
        R2 = ( R ).dot( np.linalg.inv(self.stereoRig.R) ).dot( np.linalg.inv(self.stereoRig.intrinsic2) )
        
        return R1, R2, R
    
    
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
    
    def _calculateCameraFrequency(self):
        """
        Estimate fc from system geometry and fp.
        """
        Ac = self.stereoRig.intrinsic1
        Dc = self.stereoRig.distCoeffs1
        
        Ap = self.stereoRig.intrinsic2
        R = self.stereoRig.R
        T = self.stereoRig.T
        Dp = self.stereoRig.distCoeffs2
        
        Op = -np.linalg.inv(R).dot(T)
        ObjCenter = np.array([[[0],[0],[self.lc]]], dtype=np.float32)
        
        # Send center of reference plane to projector
        pCenter, _ = cv2.projectPoints(ObjCenter, R, T, 
            self.stereoRig.intrinsic2, self.stereoRig.distCoeffs2)
        # Now we are in the projected image
        # Perfect fringe pattern. No distortion
        
        # Find two points at distance Tp (period on projector)
        halfPeriodP = (1/self.fp)/2
        
        if self.horizontal:
            topP = pCenter[0,0,1] - halfPeriodP
            bottomP = pCenter[0,0,1] + halfPeriodP
            points = np.array([[pCenter[0,0,0],topP], [pCenter[0,0,0],bottomP]])
        else:
            leftP = pCenter[0,0,0] - halfPeriodP
            rightP = pCenter[0,0,0] + halfPeriodP
            points = np.array([[leftP,pCenter[0,0,1]], [rightP,pCenter[0,0,1]]])
        
        # Un-distort points for the projector means to distort
        # as the pinhole camera model is made for cameras
        # and we are projecting back to 3D world
        distortedPoints = cv2.undistortPoints(points, np.eye(3), Dp)
                                              
        # De-project points to reference plane
        invARp = np.linalg.inv(Ap.dot(R))
        M = np.array([[self.lc-Op[2],0,Op[0]],[0,self.lc-Op[2],Op[1]], [0,0,self.lc]],dtype=np.object)
        
        wLeft = invARp.dot(np.append(distortedPoints[0,0,:], [1]))  # Top for horizontal case
        wRight = invARp.dot(np.append(distortedPoints[1,0,:], [1])) # Bottom for horizontal case
        
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
        if self.horizontal:
            Tc = ((b[0,0,1] - b[1,0,1])**2 + (b[0,0,0] - b[1,0,0])**2)/abs(b[0,0,1]-b[1,0,1])
        else:
            # Use the first Euclid theorem to get the wanted period
            Tc = ((b[0,0,0] - b[1,0,0])**2 + (b[0,0,1] - b[1,0,1])**2)/abs(b[0,0,0]-b[1,0,0])
        
        # Return frequency
        return 1/Tc    
    
    def scan(self, image, fc=None, radius_factor=0.5, roi=None, unwrappingMethod=None, plot=False):
        """
        Process an image and get the point cloud.
        
        Parameters
        ----------
        image : numpy.ndarray
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
            as argument. If None (default) `np.unwrap`is used.
            
        Returns
        -------
        Point cloud with shape (height,width,2), with height and width 
        same as the input image or selected `roi`.
        
        .. todo::
            If `roi`, `fc` and `radius_factor` are left unchanged, the
            reference image cut and filtering is uselessly done each time.
        
        """
        
        # Check that is a color image
        if image.ndim != 3:
            raise ValueError("image must be a BGR color image!")
        
        if fc is None:
            fc = self.fc
        
        if unwrappingMethod is None:
            unwrappingMethod = self.unwrapBasic
            
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
        
        if self.horizontal:
            G0 = np.fft.fft(imgR_gray, axis=0)     # FFT on y dimension
            G = np.fft.fft(imgObj_gray, axis=0)
            # Get frequencies associated to each value
            # (depends only on lenght of input, with step=1)
            freqs = np.fft.fftfreq(roi_h)
        else:
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
            
            if self.horizontal:
                plt.suptitle("Middle column absolute phase")
                # Show module of FFTs
                plt.plot(freqs[:roi_h//2], np.absolute(G0[:roi_h//2,roi_w//2-1]), label="|G0|", linestyle='--', color='red')
                plt.plot(freqs[:roi_h//2], np.absolute(G[:roi_h//2,roi_w//2-1]), label="|G|", linestyle='-', color='blue')
                # Show filtered band
                plt.axvline(x=freqs[fIndex], linestyle='-', color='black')
                plt.axvline(x=freqs[fminIndex], linestyle='dotted', color='black')
                plt.axvline(x=freqs[fmaxIndex], linestyle='dotted', color='black')
            else:
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
        
        if self.horizontal:
            # Reference image filtering
            G0[:fminIndex,:] = 0
            G0[fmaxIndex+1:,:] = 0
            g0hat = np.fft.ifft(G0,axis=0)
            # Object image filtering
            G[:fminIndex,:] = 0
            G[fmaxIndex+1:,:] = 0
            ghat = np.fft.ifft(G,axis=0)
            
        else:   
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
            phaseUnwrapped = np.unwrap(phase)
        else:
            phaseUnwrapped = unwrappingMethod(phase)
              
        if plot:
            if self.horizontal:
                plt.title("Middle column phase")
                plt.plot(np.arange(roi_h), phase[:,roi_w//2-1], label="Phase", linestyle='-.', color='red')
                plt.plot(np.arange(roi_h), phaseUnwrapped[:,roi_w//2-1], label="Unwrapped phase", linestyle='-', color='blue')
            else:
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
        
        # Two rectification homographies and new common orientation
        # were calculated during initialization
        # self.Rectify1, self.Rectify2, self.Rotation
        
        ### Get inverse common orientation and extend to 4x4 transform
        R_inv = np.linalg.inv(self.Rotation)
        R_inv = np.hstack( ( np.vstack( (R_inv,np.zeros((1,3))) ), np.zeros((4,1)) ) )
        R_inv[3,3] = 1
        
        ### Find k values from central stripe
        # Triangulation can be done directly between left and right red stripe
        # and then compare with the phase value, but, as a quicker method
        # we find approximated A and H points over projector image plane
        # and count the integer periods of shift between the two
        
        # Find central line on the undistorted object image
        objStripe = ss.active.findCentralStripe(imgObj, self.stripeColor,
                        self.stripeThreshold, self.horizontal)
        
        # Find integer indexes (round half down)
        # Accept loss of precision as k values must be rounded to integers
        cam_indexes = np.ceil(objStripe-0.5).astype(np.int) # As (x,y)
        
        pointA = projCoords[cam_indexes[:,1],cam_indexes[:,0]] # As (x,y)
        
        if self.horizontal:
            # Find Hy and Ay coords and calculate vector k
            pointA_x_rounded = np.ceil(pointA[:,0]-0.5).astype(np.int) # Round half down
            pointH_y = self.fringeStripe[pointA_x_rounded,1] # Get all y stripe coords
            theta = phaseUnwrapped[cam_indexes[:,1],cam_indexes[:,0]]
            k = (pointH_y - pointA[:,1])*self.fp - theta/(2*np.pi)
            k = np.rint(k).reshape(1,-1) # Banker's rounding + appropriate reshape
        else:
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
        
        if self.horizontal:
            Yh = Ya + phaseUnwrapped/(2*np.pi*self.fp)
            # Find x coord on epipolar line
            Xh = ( (Yh-ep[1])/(Ya-ep[1]) )*(Xa-ep[0]) + ep[0]
        else:
            Xh = Xa + phaseUnwrapped/(2*np.pi*self.fp)
            # Find y coord on epipolar line
            Yh = ( (Xh-ep[0])/(Xa-ep[0]) )*(Ya-ep[1]) + ep[1]
            
        # Desidered point is H(Xh,Yh)
        H = np.hstack((Xh,Yh)).reshape(-1,1,2).astype(np.float64)
        
        # Apply lens distortion to H (as it's a projector)
        # A projector is considered as an inversed pinhole camera
        # H is on the original imgFringe. Passing through the projector lenses,
        # it gets distortion, so it does not coincide with real world point.
        # Remove intrinsic, undistort and put same intrinsic back.
        H = cv2.undistortPoints(H, Ap, Dp, P=Ap)
        
        
        ### Triangulation
        
        # Build grid of indexes and apply rectification (undistorted camera points)
        pc = np.mgrid[0:widthC,0:heightC].T
        pc = pc[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w].reshape(-1,1,2).astype(np.float64)
        # Consider pixel center (see also projCoords in self._getProjectorMapping)
        pc = pc + 0.5
        pc = cv2.perspectiveTransform(pc, self.Rectify1).reshape(-1,2) # Apply rectification
        # Add ones as third coordinate
        pc = np.hstack( (pc,np.ones((roi_w*roi_h,1),dtype=np.float64)) )
        
        # Apply rectification to projector points.
        # Rectify2 cancels the intrinsics and applies new rotation.
        # No new intrinsics here.
        pp = cv2.perspectiveTransform(H, self.Rectify2)
        pp = pp.reshape(-1,2)
        
        # Get world points
        disparity = np.abs(pp[:,[0]] - pc[:,[0]])
        pw = self.stereoRig.getBaseline()*(pc/disparity)
        
        # Cancel common orientation applied to first camera
        # to bring points into camera coordinate system
        finalPoints = cv2.perspectiveTransform(pw.reshape(-1,1,3), R_inv)
        
        # Reshape
        finalPoints = finalPoints.reshape(roi_h,roi_w,3)
            
        return finalPoints
