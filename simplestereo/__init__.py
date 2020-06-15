"""
simplestereo
============
Common classes and functions.


.. todo::
    - Add new rig class for structured light (projector + camera)
    - Add new rig class for uncalibrated stereo
"""
import json
import numpy as np
import cv2

from . import calibration
from . import rectification
from . import passive
from . import postprocessing
from . import utils



class StereoRig:
    """ 
    Keep together and manage all parameters of a calibrated stereo rig.
    
    The essential E and fundamental F matrices are optional as they are not always available.
    They may be computed from camera parameters, if needed.
        
    Parameters
    ----------
    res1, res2 : tuple
        Resolution of camera as (width, height)  
    cameraMatrix1, cameraMatrix2 : numpy.ndarray
        3x3 intrinsic camera matrix in the form [[fx, 0, tx], [0, fy, ty], [0, 0, 1]].
    distCoeffs1, distCoeffs2 : list or numpy.ndarray
        List of distortion coefficients of 4, 5, 8, 12 or 14 elements (refer to OpenCV documentation).
    R : numpy.ndarray
        Rotation matrix between the 1st and the 2nd camera coordinate systems as numpy.ndarray.
    T : numpy.ndarray
        Translation vector between the coordinate systems of the cameras as numpy.ndarray.
    E : numpy.ndarray, optional
        Essential matrix as numpy.ndarray (default None) .
    F : numpy.ndarray, optional
        Fundamental matrix as numpy.ndarray (default None).
    reprojectionError : float, optional
        Total reprojection error resulting from calibration (default None).
    
    
    .. note:: 
        This class follows OpenCV convention to set the origin of the world coordinate system into the first camera.
        Hence the first camera extrinsics parameters will always be identity matrix rotation and zero translation.
        If your world coordinate system is placed into a camera, you must convert it to use this class
        (see :func:`ss.utils.moveExtrinsicOriginToFirstCamera`).
    
    """  
    def __init__(self, res1, res2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E=None, F=None, reprojectionError=None):
        self.res1 = res1
        self.res2 = res2
        self.cameraMatrix1 = np.array(cameraMatrix1)
        self.cameraMatrix2 = np.array(cameraMatrix2)
        self.R = np.array(R)
        self.T = np.array(T).reshape((-1,1))              
        self.distCoeffs1 = np.array(distCoeffs1) if any(distCoeffs1) else np.zeros(5) # Convert to numpy.ndarray
        self.distCoeffs2 = np.array(distCoeffs2) if any(distCoeffs2) else np.zeros(5)
        self.F = np.array(F) if F is not None else None
        self.E = np.array(E) if E is not None else None
        self.reprojectionError = reprojectionError
        
    @classmethod 
    def fromFile(cls, filepath):
        """
        Alternative initialization of StereoRig object from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path of the JSON file containing saved parameters of the stereo rig.
        
        Returns
        -------
        StereoRig
            An object of StereoRig class.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        res1 = tuple(data.get('res1'))
        res2 = tuple(data.get('res2'))
        cameraMatrix1 = np.array(data.get('cameraMatrix1'))
        cameraMatrix2 = np.array(data.get('cameraMatrix2'))
        R = np.array(data.get('R'))
        T = np.array(data.get('T')).reshape((-1,1))              
        distCoeffs1 = np.array(data.get('distCoeffs1'))
        distCoeffs2 = np.array(data.get('distCoeffs2'))
        F = np.array(data.get('F')) if data.get('F') else None
        E = np.array(data.get('E')) if data.get('E') else None
        reprojectionError = data.get('reprojectionError')
        
        return cls(res1, res2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, reprojectionError)
        
    def save(self, filepath):
        """
        Save configuration to JSON file.
        
        Save the current stereo rig configuration to a JSON file that can be loaded later.
        
        Parameters
        ----------
        filepath : str
            Path where to save the JSON file containing the parameters of the stereo rig.
        """
        with open(filepath, 'w') as f:
            out = {}
            out['res1'] = self.res1
            out['res2'] = self.res2
            out['cameraMatrix1'] = self.cameraMatrix1.tolist()
            out['cameraMatrix2'] = self.cameraMatrix2.tolist()
            out['R'] = self.R.tolist()
            out['T'] = self.T.tolist()
            out['distCoeffs1'] = self.distCoeffs1.tolist()
            out['distCoeffs2'] = self.distCoeffs2.tolist()
            if self.F is not None:
                out['F'] = self.F.tolist()
            if self.E is not None:
                out['E'] = self.E.tolist()
            if self.reprojectionError:
                out['reprojectionError'] = self.reprojectionError
            json.dump(out, f)
            
    
    def getCenters(self):
        """
        Calculate camera centers in world coordinates.
        
        Returns
        -------
        numpy.ndarray
            3D coordinates of first camera center
        numpy.ndarray
            3D coordinates of second camera center
        """
        Po1, Po2 = self.getProjectionMatrices()
        #C1 = np.zeros(3)    # World origin is set in camera 1
        C1 = -np.linalg.inv(Po1[:,:3]).dot(Po1[:,3])
        C2 = -np.linalg.inv(Po2[:,:3]).dot(Po2[:,3])
        return C1, C2
    
    def getBaseline(self):
        """
        Calculate the norm of the vector from camera 1 to camera 2.
        
        Returns
        -------
        float
            Length of the baseline in world units.
        """
        C1, C2 = self.getCenters()
        return np.linalg.norm(C2) # No need to do C2 - C1 as C1 is always zero (origin of world system)
    
    
    def getProjectionMatrices(self):
        """
        Calculate the projection matrices of camera 1 and camera 2.
        
        Returns
        -------
        numpy.ndarray
            The 3x4 projection matrix of the first camera.
        numpy.ndarray
            The 3x4 projection matrix of the second camera.
        """
        Po1 = np.hstack( (self.cameraMatrix1, np.zeros((3,1))) )
        Po2 = self.cameraMatrix2.dot( np.hstack( (self.R, self.T) ) )
        return Po1, Po2
    
    def getFundamentalMatrix(self):
        """
        Returns the fundamental matrix F.
        
        If not set, F is computed from projection matrices using 
        :func:`simplestereo.calibration.getFundamentalMatrixFromProjections`.
        
        Returns
        -------
        F : numpy.ndarray
            The 3x3 fundamental matrix.
        
        Notes
        -----
        The fundamental matrix has always a free scaling factor.
        """
        if self.F is None:   # If F is not set, calculate it and update the object data.
            P1, P2 = self.getProjectionMatrices()
            self.F = calibration.getFundamentalMatrixFromProjections(P1,P2)
                
        return self.F
    
    def getEssentialMatrix(self):
        """
        Returns the essential matrix E.
        
        If not set, E is computed from the fundamental matrix F and the camera matrices.
        
        Returns
        -------
        E : numpy.ndarray
            The 3x3 essential matrix.
        
        Notes
        -----
        The essential matrix has always a free scaling factor.
        """
        if True or self.E is None:  # If E is not set, calculate it and update the object data.
            F = self.getFundamentalMatrix()
            self.E = self.cameraMatrix2.T.dot(F).dot(self.cameraMatrix1)
            
        return self.E
    
    def undistortImages(self, img1, img2, changeCameras=False, alpha=1, destDims=None, centerPrincipalPoint=False):
        """
        Undistort two given images of the stereo rig.
        
        This method wraps ``cv2.getOptimalNewCameraMatrix()`` followed by ``cv2.undistort()`` for both images.
        If changeCameras is False, original camera matrices are used, otherwise all the parameters of ``cv2.getOptimalNewCameraMatrix()`` 
        are considered when undistorting the images.
        
        Parameters
        ----------
        img1, img2 : cv2.Mat
            A couple of OpenCV images taken with the stereo rig (ordered).
        changeCameras : bool
            If False (default) the original camera matrices are used and all the following parameters are skipped.
            If True, new camera matrices are computed with the given parameters.
        alpha : float
            Scaling parameter for both images.
            If alpha=0, it returns undistorted image with minimum unwanted pixels
            (so it may even remove some pixels at image corners). If alpha=1, all pixels are retained 
            with some extra black images. Values in the middle are accepted too (default to 1).
        destDims : tuple, optional
            Resolution of destination images as (width, height) tuple (default to first image resolution).
        centerPrincipalPoint : bool
            If True the principal point is centered within the images (default to False).
            
        Returns
        -------
        img1_undist, img2_undist : cv2.Mat
            The undistorted images.
        cameraMatrixNew1, cameraMatrixNew2 : numpy.ndarray
            If *changeCameras* is set to True, the new camera matrices are returned too.
        
        See Also
        --------
        cv2.getOptimalNewCameraMatrix
        cv2.undistort
        """
        if changeCameras:   # Change camera matrices
            cameraMatrixNew1, _ = cv2.getOptimalNewCameraMatrix(self.cameraMatrix1, self.distCoeffs1, self.res1, alpha, destDims, centerPrincipalPoint)
            cameraMatrixNew2, _ = cv2.getOptimalNewCameraMatrix(self.cameraMatrix2, self.distCoeffs2, self.res2, alpha, destDims, centerPrincipalPoint)
            
            img1_undist = cv2.undistort(img1, self.cameraMatrix1, self.distCoeffs1, None, cameraMatrixNew1)
            img2_undist = cv2.undistort(img2, self.cameraMatrix2, self.distCoeffs2, None, cameraMatrixNew2)
            
            return img1_undist, img2_undist, cameraMatrixNew1, cameraMatrixNew2
        
        else:   # Use original camera matrices
            img1_undist = cv2.undistort(img1, self.cameraMatrix1, self.distCoeffs1, None, None)
            img2_undist = cv2.undistort(img2, self.cameraMatrix2, self.distCoeffs2, None, None)
            
            return img1_undist, img2_undist
        
        
        
        
        
class RectifiedStereoRig(StereoRig):
    """
    Keep together and manage all parameters of a calibrated and rectified stereo rig.
    
    It includes all the parameters of StereoRig plus two rectifying homographies. Differently from OpenCV,
    here the rectifying *homographies* (pixel domain) are taken as input, the ones commonly referred in literature, and
    **not** the rectification transformation in the object space.
    
    Parameters
    ----------
    rectHomography1, rectHomography1 : np.array
        3x3 matrices representing rectification homographies computed with one of the rectification methods.
    
    StereoRig
        A StereoRig object or, *alternatively*, all the parameters of :meth:`simplestereo.StereoRig` (in the same order).
    """
    def __init__(self, rectHomography1, rectHomography2, *args):
        
        self.rectHomography1 = rectHomography1
        self.rectHomography2 = rectHomography2
        self.K1 = None                                       # Fitting affine transformations
        self.K2 = None                                       # aka new camera matrices after rectification
        
        if isinstance(args[0], StereoRig):                  # Extend unpacking a StereoRig object 
            r = args[0]
            super(RectifiedStereoRig, self).__init__(r.res1, r.res2, r.cameraMatrix1, r.distCoeffs1, r.cameraMatrix2, r.distCoeffs2, r.R, r.T, r.E, r.reprojectionError)
        else:                                               # Or use all the parameters
            super(RectifiedStereoRig, self).__init__(*args)
        
        self.computeRectificationMaps()
        
    @classmethod 
    def fromFile(cls, filepath):
        """
        Alternative initialization of StereoRigRectified object from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path of the JSON file containing saved parameters of the stereo rig.
        
        Returns
        -------
        StereoRigRectified
            An object of StereoRigRectified class.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        rectHomography1 = np.array(data.get('rectHomography1'))
        rectHomography2 = np.array(data.get('rectHomography2'))
        res1 = tuple(data.get('res1'))
        res2 = tuple(data.get('res2'))
        cameraMatrix1 = np.array(data.get('cameraMatrix1'))
        cameraMatrix2 = np.array(data.get('cameraMatrix2'))
        R = np.array(data.get('R'))
        T = np.array(data.get('T'))              
        distCoeffs1 = np.array(data.get('distCoeffs1'))
        distCoeffs2 = np.array(data.get('distCoeffs2'))
        F = np.array(data.get('F'))
        E = np.array(data.get('E'))
        reprojectionError = data.get('reprojectionError')
        
        return cls(rectHomography1, rectHomography2, res1, res2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, reprojectionError)
        
    def save(self, filepath):
        """
        Save configuration to JSON file.
        
        Save the current stereo rig configuration to a JSON file that can be loaded later.
        
        Parameters
        ----------
        filepath : str
            Path where to save the JSON file containing the parameters of the stereo rig.
        """
        with open(filepath, 'w') as f:
            out = {}
            out['rectHomography1'] = self.rectHomography1.tolist()
            out['rectHomography2'] = self.rectHomography2.tolist()
            out['res1'] = self.res1
            out['res2'] = self.res2
            out['cameraMatrix1'] = self.cameraMatrix1.tolist()
            out['cameraMatrix2'] = self.cameraMatrix2.tolist()
            out['R'] = self.R.tolist()
            out['T'] = self.T.tolist()
            out['distCoeffs1'] = self.distCoeffs1.tolist()
            out['distCoeffs2'] = self.distCoeffs2.tolist()
            if self.F is not None:
                out['F'] = self.F.tolist()
            if self.E is not None:
                out['E'] = self.E.tolist()
            if self.reprojectionError:
                out['reprojectionError'] = self.reprojectionError
            json.dump(out, f)
    
    
    def getRectificationTransformations(self):
        """
        Get raw rectification 3x3 matrices without any affine trasformation.
        
        Returns
        -------
        rectHomography1, rectHomography2 : numpy.ndarray
            3x3 rectification transformations for camera 1 and camera 2.
        """
        return self.rectHomography1, self.rectHomography2
    
    
    def computeRectificationMaps(self, destDims=None, zoom=1):
        """
        Compute the two maps to undistort and rectify the stereo pair.
        
        This method wraps ``cv2.initUndistortRectifyMap()`` plus a custom fitting algorithm to keep image within dimensions. 
        It modifies the original camera matrix applying affine transformations (x-y scale and translation, shear (x axis only)) 
        without losing rectification. The two new maps are stored internally.
        This method is called in the constructor with default parameters and can be called later to change its settings.
        
        Parameters
        ----------
        alpha : float
            Scaling parameter for both images. If alpha=0, it returns undistorted image with minimum unwanted pixels
            (so it may even remove some pixels at image corners). If alpha=1, all pixels are retained 
            with some extra black images. Values in the middle are accepted too (default to 1).
        destDims: tuple, optional
            Resolution of destination images as (width, height) tuple (default to first image resolution).
        centerPrincipalPoint : bool
            If True the principal point is centered within the images (default to False).
        
        Returns
        -------
        None
        
        Notes
        -----
        OpenCV uses *rectification transformation in the object space (3x3 matrix)*, but most of the papers provide algorithms
        to compute the homography to be applied to the *image* in a pixel domain, not a rotation matrix R in 3D space.
        This library always refers to rectification transform as the ones in pixel domain.
        To adapt it to be used with OpenCV the transformation to be used in :func:`cv2.initUndistortRectifyMap()` (and other functions)
        is given by ``rectHomography.dot(cameraMatrix)``.
        For each camera, the function computes homography H as the rectification transformation.
        
        ..todo::
            destDims needs to be considered in computing the Q matrix for 3D reconstruction.
        """
        if destDims is None:
            destDims = self.res1
            
        # Find fitting matrix
        K1, K2 = rectification.getFittingMatrix(self.cameraMatrix1, self.cameraMatrix2, self.rectHomography1, self.rectHomography2, self.res1, self.res2, self.distCoeffs1, self.distCoeffs2, destDims, zoom)
        
        # Build rectification transforms from homographies (see cv2.initUndistortRectifyMap documentation)
        R1 = self.rectHomography1.dot(self.cameraMatrix1)
        R2 = self.rectHomography2.dot(self.cameraMatrix2)
        
        # Recompute final maps considering fitting transformations too
        mapx1_new, mapy1_new = cv2.initUndistortRectifyMap(self.cameraMatrix1, self.distCoeffs1, R1, K1, destDims, cv2.CV_32FC1)
        mapx2_new, mapy2_new = cv2.initUndistortRectifyMap(self.cameraMatrix2, self.distCoeffs2, R2, K2, destDims, cv2.CV_32FC1)
        self.mapx1 = mapx1_new
        self.mapy1 = mapy1_new
        self.mapx2 = mapx2_new
        self.mapy2 = mapy2_new
        self.K1 = K1
        self.K2 = K2
        
        
    def rectifyImages(self, img1, img2, interpolation=cv2.INTER_LINEAR):
        """
        Undistort, rectify and apply affine transformation to a couple of images coming from the stereo rig.
        
        *img1* and *img2* must be provided as in calibration (es. img1 is the left image, img2 the right one).
        This method is wraps ``cv2.remap()`` for two images of the stereo pair. The maps used are computed by
        :meth:`computeRectificationMaps` during initialization (with default parameters). 
        :meth:`computeRectificationMaps` can be called before calling this method to change mapping settings (e.g. final resolution).
        
        Parameters
        ----------
        img1, img2 : cv2.Mat
            A couple of OpenCV images taken with the stereo rig (ordered).
        interpolation : int, optional
            OpenCV *InterpolationFlag*. The most common are ``cv2.INTER_NEAREST``, ``cv2.INTER_LINEAR`` (default) and ``cv2.INTER_CUBIC``.
        
        Returns
        -------
        img1_rect, img2_rect : cv2.Mat
            The undistorted images.
        """
        img1_rect = cv2.remap(img1, self.mapx1, self.mapy1, interpolation);
        img2_rect = cv2.remap(img2, self.mapx2, self.mapy2, interpolation);
        
        return img1_rect, img2_rect
    
    
    def getQ(self):
        """
        Get the Q matrix to be used as input of ``cv2.reprojectImageTo3D``, together with the disparity map.
        
        After rectification a common camera matrix K is applied to both images. The baseline remains the same.
        
        Returns
        -------
        numpy.ndarray
            A 4x4 matrix.
        """
        b = self.getBaseline()
        Q = np.eye(4)
        Q[2,2] = 0
        Q[0,3] = -self.K1[0,2]                  # -cx
        Q[1,3] = -self.K1[1,2]                  # -cy
        Q[2,3] = self.K1[0,0]                   # fx
        Q[3,2] = -1/b                           #-1/Tx
        Q[3,3] = (self.K1[0,2]-self.K2[0,2])/b  #cx-cx'/Tx
        
        return Q    
