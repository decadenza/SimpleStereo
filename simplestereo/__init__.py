"""
simplestereo
============
Common classes and functions.


Documentation DOCSTRING follows numpy-style wherever possible.
See https://numpydoc.readthedocs.io/en/latest/format.html

.. todo::
    - Add new rig class for structured light (projector + camera).
    - Add new rig class for uncalibrated stereo.
"""
import json
import numpy as np
import cv2

from . import calibration
from . import rectification
from . import passive
from . import active
from . import unwrapping
from . import points
from . import utils


### VERSION INFO
import pkg_resources # part of setuptools
__version__ = pkg_resources.require("SimpleStereo")[0].version


class StereoRig:
    """ 
    Keep together and manage all parameters of a calibrated stereo rig.
    
    The essential E and fundamental F matrices are optional as they are not always available.
    They may be computed from camera parameters, if needed.
        
    Parameters
    ----------
    res1, res2 : tuple
        Resolution of camera as (width, height)  
    intrinsic1, intrinsic2 : numpy.ndarray
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
        (see :func:`simplestereo.utils.moveExtrinsicOriginToFirstCamera`).
    
    """  
    def __init__(self, res1, res2, intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F=None, E=None, reprojectionError=None):
        self.res1 = res1
        self.res2 = res2
        self.intrinsic1 = np.array(intrinsic1)
        self.intrinsic2 = np.array(intrinsic2)
        self.distCoeffs1 = np.array(distCoeffs1) if distCoeffs1 is not None else np.zeros(5) # Convert to numpy.ndarray
        self.distCoeffs2 = np.array(distCoeffs2) if distCoeffs2 is not None else np.zeros(5)
        self.R = np.array(R)
        self.T = np.array(T).reshape((-1,1))              
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
        intrinsic1 = np.array(data.get('intrinsic1'))
        intrinsic2 = np.array(data.get('intrinsic2'))
        R = np.array(data.get('R'))
        T = np.array(data.get('T')).reshape((-1,1))              
        distCoeffs1 = np.array(data.get('distCoeffs1'))
        distCoeffs2 = np.array(data.get('distCoeffs2'))
        F = np.array(data.get('F')) if data.get('F') else None
        E = np.array(data.get('E')) if data.get('E') else None
        reprojectionError = data.get('reprojectionError')
        
        return cls(res1, res2, intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F, E, reprojectionError)
    
        
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
            out['intrinsic1'] = self.intrinsic1.tolist()
            out['intrinsic2'] = self.intrinsic2.tolist()
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
        
        Anyway first camera will always be centered in zero (returned anyway).
        
        Returns
        -------
        numpy.ndarray
            3D coordinates of first camera center (always zero).
        numpy.ndarray
            3D coordinates of second camera center.
        """
        Po1, Po2 = self.getProjectionMatrices()
        C1 = np.zeros(3)    # World origin is set in camera 1
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
        Po1 = np.hstack( (self.intrinsic1, np.zeros((3,1))) )
        Po2 = self.intrinsic2.dot( np.hstack( (self.R, self.T) ) )
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
            #P1, P2 = self.getProjectionMatrices()
            #self.F = calibration.getFundamentalMatrixFromProjections(P1,P2)
            # Alternative formula by
            # Multiple View Geometry in Computer Vision, by Richard Hartley and Andrew Zisserman
            vv = utils.getCrossProductMatrix(self.intrinsic1.dot(self.R.T).dot(self.T))
            self.F = (np.linalg.inv(self.intrinsic2).T).dot(self.R).dot(self.intrinsic1.T).dot(vv)
                
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
            self.E = self.intrinsic2.T.dot(F).dot(self.intrinsic1)
            
        return self.E
    
    
    def undistortImages(self, img1, img2, changeCameras=False, alpha=1, destDims=None, centerPrincipalPoint=False):
        """
        Undistort two given images of the stereo rig.
        
        This method wraps `cv2.getOptimalNewCameraMatrix()` followed
        by `cv2.undistort()` for both images.
        If changeCameras is False, original camera matrices are used,
        otherwise all the parameters of
        `cv2.getOptimalNewCameraMatrix()` are considered when 
        undistorting the images.
        
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
            cameraMatrixNew1, _ = cv2.getOptimalNewCameraMatrix(self.intrinsic1, self.distCoeffs1, self.res1, alpha, destDims, centerPrincipalPoint)
            cameraMatrixNew2, _ = cv2.getOptimalNewCameraMatrix(self.intrinsic2, self.distCoeffs2, self.res2, alpha, destDims, centerPrincipalPoint)
            
            img1_undist = cv2.undistort(img1, self.intrinsic1, self.distCoeffs1, None, cameraMatrixNew1)
            img2_undist = cv2.undistort(img2, self.intrinsic2, self.distCoeffs2, None, cameraMatrixNew2)
            
            return img1_undist, img2_undist, cameraMatrixNew1, cameraMatrixNew2
        
        else:   # Use original camera matrices
            img1_undist = cv2.undistort(img1, self.intrinsic1, self.distCoeffs1, None, None)
            img2_undist = cv2.undistort(img2, self.intrinsic2, self.distCoeffs2, None, None)
            
            return img1_undist, img2_undist
          
        
class RectifiedStereoRig(StereoRig):
    """
    Keep together and manage all parameters of a calibrated and rectified stereo rig.
    
    It includes all the parameters of StereoRig plus two rectifying homographies. Differently from OpenCV,
    here the rectifying *homographies* (pixel domain) are taken as input, the ones commonly referred in literature, and
    **not** the rectification transformation in the object space.
    
    Parameters
    ----------
    Rcommon : np.array
        3x3 matrices representing the new common camera orientation
        after rectification.
    rectHomography1, rectHomography2 : np.array
        3x3 rectification homographies.
    StereoRig:
        A StereoRig object or, *alternatively*, all the parameters of
        :meth:`simplestereo.StereoRig` (in the same order).
    """
    def __init__(self, Rcommon, rectHomography1, rectHomography2, *args):
        
        self.Rcommon = Rcommon                  # Common camera orientation
        self.rectHomography1 = rectHomography1  # Rectification homographies
        self.rectHomography2 = rectHomography2
        
        # Final intrinsic matrices that keep track of all affine transformations applied, are
        # calculated in self.computeRectificationMaps() considering destination dimensions and zoom.
        # N.B. These will be needed for depth reconstruction!
        self.K1 = None
        self.K2 = None
        
        if isinstance(args[0], StereoRig):                  # Extend unpacking a StereoRig object 
            r = args[0]
            super(RectifiedStereoRig, self).__init__(r.res1, r.res2, r.intrinsic1, r.intrinsic2, r.distCoeffs1, r.distCoeffs2, r.R, r.T, r.F, r.E, r.reprojectionError)
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
        
        Rcommon = np.array(data.get('Rcommon'))
        rectHomography1 = np.array(data.get('rectHomography1'))
        rectHomography2 = np.array(data.get('rectHomography2'))
        res1 = tuple(data.get('res1'))
        res2 = tuple(data.get('res2'))
        intrinsic1 = np.array(data.get('intrinsic1'))
        intrinsic2 = np.array(data.get('intrinsic2'))
        R = np.array(data.get('R'))
        T = np.array(data.get('T'))              
        distCoeffs1 = np.array(data.get('distCoeffs1'))
        distCoeffs2 = np.array(data.get('distCoeffs2'))
        F = np.array(data.get('F'))
        E = np.array(data.get('E'))
        reprojectionError = data.get('reprojectionError')
        
        return cls(Rcommon, rectHomography1, rectHomography2, res1, res2, intrinsic1, intrinsic2, distCoeffs1, distCoeffs2, R, T, F, E, reprojectionError)
    
        
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
            out['Rcommon'] = self.Rcommon.tolist()
            out['rectHomography1'] = self.rectHomography1.tolist()
            out['rectHomography2'] = self.rectHomography2.tolist()
            out['res1'] = self.res1
            out['res2'] = self.res2
            out['intrinsic1'] = self.intrinsic1.tolist()
            out['intrinsic2'] = self.intrinsic2.tolist()
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
    
    
    def getRectifiedProjectionMatrices(self):
        """
        Calculate the projection matrices of camera 1 and camera 2 after rectification.
        
        New projection matrices, after rectification, share the same orientation `Rcommon`,
        have only one horizontal displacement (the baseline) and
        have new intrinsics (`K1` and `K2`) that depends on all the rigid manipulation done after rectification.
        
        Returns
        -------
        numpy.ndarray
            The 3x4 projection matrix of the first camera.
        numpy.ndarray
            The 3x4 projection matrix of the second camera.
        """
        C1, C2 = self.getCenters()
        P1 = self.K1.dot(self.Rcommon).dot( np.hstack( (np.eye(3), -C1[:,None]) ) )
        P2 = self.K2.dot(self.Rcommon).dot( np.hstack( (np.eye(3), -C2[:,None]) ) )
        return P1, P2  
    
    
    def computeRectificationMaps(self, destDims=None, zoom=1):
        """
        Compute the two maps to undistort and rectify the stereo pair.
        
        This method wraps ``cv2.initUndistortRectifyMap()`` plus a custom fitting algorithm to keep image within dimensions. 
        It modifies the original camera matrix applying affine transformations (x-y scale and translation, shear (x axis only)) 
        without losing rectification. The two new maps are stored internally.
        This method is called in the constructor with default parameters and can be called later to change its settings.
        
        Parameters
        ----------
        destDims: tuple, optional
            Resolution of destination images as (width, height) tuple (default to first image resolution).
        zoom: float, optional
            Zoom on the final images. Default to 1.
        
        Returns
        -------
        None
        
        Notes
        -----
        OpenCV uses *rectification transformation in the object space (3x3 matrix)*, but most of the papers provide algorithms
        to compute the homography to be applied to the *image* in a pixel domain, not a rotation matrix R in 3D space.
        This library always refers to rectification transform as the ones in pixel domain.
        To adapt it to be used with OpenCV the transformation to be used in :func:`cv2.initUndistortRectifyMap()` (and other functions)
        is given by `rectHomography.dot(cameraMatrix)`.
        For each camera, the function computes homography H as the rectification transformation.
        """
        if destDims is None:
            destDims = self.res1
        
        # Find fitting matrices, as additional correction of the new camera matrices (if any).
        # Useful e.g. to change destination image resolution or zoom.
        Fit1, Fit2 = rectification.getFittingMatrices(self.intrinsic1, self.intrinsic2, self.rectHomography1, self.rectHomography2, self.res1, self.res2, self.distCoeffs1, self.distCoeffs2, destDims, zoom)
        
        # Isolate affine transformation applied after rectification
        # These would be the FINAL new camera intrinsics (needed for 3D reconstrunction)
        self.K1 = Fit1.dot(self.rectHomography1).dot( self.intrinsic1 ).dot(self.Rcommon.T)
        self.K2 = Fit2.dot(self.rectHomography2).dot( self.intrinsic2.dot(self.R) ).dot(self.Rcommon.T)
        
        # OpenCV requires the final rotations applied
        R1 = self.Rcommon
        R2 = self.Rcommon.dot(self.R.T)
        
        # Recompute final maps considering fitting transformations too
        #P1, P2 = self.getRectifiedProjectionMatrices()
        self.mapx1, self.mapy1 = cv2.initUndistortRectifyMap(self.intrinsic1, self.distCoeffs1, R1, self.K1, destDims, cv2.CV_32FC1)
        self.mapx2, self.mapy2 = cv2.initUndistortRectifyMap(self.intrinsic2, self.distCoeffs2, R2, self.K2, destDims, cv2.CV_32FC1)
        
        
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
    
    
    def get3DPoints(self, disparityMap):
        """
        Get the 3D points in the space from the disparity map.
        
        If the calibration was done with real world units (e.g. millimeters),
        the output would be in the same units. The world origin will be in the
        left camera.
        
        Parameters
        ----------
        disparityMap : numpy.ndarray
            A dense disparity map having same height and width of images.
            
        Returns
        -------
        numpy.ndarray
            Array of points having shape *(height,width,3)*, where at each y,x coordinates
            a *(x,y,z)* point is associated.
        
        """
        height, width = disparityMap.shape[:2]
        
        # Build the Q matrix as OpenCV requirement
        # to be used as input of ``cv2.reprojectImageTo3D``
        # We need to cancel the final intrinsics (contained in self.K1
        # and self.K2)
        
        # IMPLEMENTATION NOTES
        # fx and fy are assumed the same for left and right (after
        # rectification, they should)
        # Accepts different x-shear terms (generally not used)
        # cx1 is not the same of cx2
        # cy1 is equal cy2 (as images are rectified)
        
        b   = self.getBaseline()
        fx  = self.K1[0,0]
        fy  = self.K2[1,1]
        cx1 = self.K1[0,2]
        cx2 = self.K2[0,2]
        a1  = self.K1[0,1]
        a2  = self.K2[0,1]
        cy  = self.K1[1,2]
        
        Q = np.eye(4, dtype='float64')
        
        Q[0,1] = -a1/fy
        Q[0,3] = a1*cy/fy - cx1
        
        Q[1,1] = fx/fy
        Q[1,3] = -cy*fx/fy
                                 
        Q[2,2] = 0
        Q[2,3] = -fx
        
        Q[3,1] = (a2-a1)/(fy*b)
        Q[3,2] = 1/b                        
        Q[3,3] = ((a1-a2)*cy+(cx2-cx1)*fy)/(fy*b)    
        
        
        return cv2.reprojectImageTo3D(disparityMap, Q)


class StructuredLightRig(StereoRig):
    """
    StereoRig child class with structured light methods.
    """
    def __init__(self, r):
        if isinstance(r, StereoRig):                  # Extend unpacking a StereoRig object 
            super(StructuredLightRig, self).__init__(r.res1, r.res2, r.intrinsic1, r.intrinsic2, r.distCoeffs1, r.distCoeffs2, r.R, r.T, r.F, r.E, r.reprojectionError)
        else:
            raise ValueError("Invalid argument!")
                
        self._computeMatrices()
    
    
    def _computeMatrices(self):
        self.R1, self.R2, self.R = rectification._lowLevelRectify(self)    
        ### Get inverse common orientation and extend to 4x4 transform
        R_inv = np.linalg.inv(self.R)
        R_inv = np.hstack( ( np.vstack( (R_inv,np.zeros((1,3))) ), np.zeros((4,1)) ) )
        R_inv[3,3] = 1
        self.R_inv = R_inv
    
    def fromFile(self):
        return StructuredLightRig(StereoRig.fromFile(self))
        
    def triangulate(self, camPoints, projPoints):
        """
        Given camera-projector correspondences, proceed with
        triangulation.
        
        Parameters
        ----------
        camPoints, projPoints: numpy.ndarray
            Ordered corresponding coordinates as (x, y) couples from 
            camera and projector. Last dimension must be 2.
            Camera points must be already undistorted.
        
        Returns
        -------
        3D coordinates with shape (-1, 1, 3).
        """
        
        pc = camPoints.reshape(-1,1,2)
        pp = projPoints.reshape(-1,1,2)
        
        pc = cv2.perspectiveTransform(pc, self.R1).reshape(-1,2) # Apply rectification
        # Add ones as third coordinate
        pc = np.hstack( (pc,np.ones((pc.shape[0],1),dtype=np.float64)) )
        
        # *Apply* lens distortion to H.
        # A projector is considered as an inversed pinhole camera (and so are
        # the distortion coefficients)
        # H is on the original imgFringe. Passing through the projector lenses,
        # it gets distortion, so it does not coincide with real world point.
        # But we want rays going exactly towards world points.
        # Remove intrinsic, undistort and put same intrinsic back.
        pp = cv2.undistortPoints(pp, self.intrinsic2, self.distCoeffs2, P=self.intrinsic2)
        # Apply rectification to projector points.
        # Rectify2 cancels the intrinsic and applies new rotation.
        # No new intrinsics here.
        pp = cv2.perspectiveTransform(pp, self.R2).reshape(-1,2)
        
        # Get world points
        disparity = np.abs(pp[:,[0]] - pc[:,[0]])
        finalPoints = self.getBaseline()*(pc/disparity)
        
        # Cancel common orientation applied to first camera
        # to bring points into camera coordinate system
        # NOT NEEDED See `rectification._lowLevelRectify` 
        finalPoints = cv2.perspectiveTransform(finalPoints.reshape(-1,1,3), self.R_inv)
        
        return finalPoints
    
    
    def undistortCameraImage(self, imgObj):
        """
        Undistort camera image.
        
        Parameters
        ----------
        imgObj : numpy.ndarray
            Camera image.
        
        Returns
        -------
        Undistorted image.
        """
        return cv2.undistort(imgObj, self.intrinsic1, self.distCoeffs1)
        
