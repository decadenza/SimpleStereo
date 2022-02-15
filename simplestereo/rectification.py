"""
rectification
=============
Contains different rectification algorithms.
"""
import math
import warnings

import numpy as np
import cv2
import scipy.optimize as optimize
from scipy.linalg import null_space, cholesky

import simplestereo as ss


def getFittingMatrices(intrinsicMatrix1, intrinsicMatrix2, H1, H2, dims1, dims2, distCoeffs1=None, distCoeffs2=None, destDims=None, zoom=1):
    """
    Compute affine tranformation to fit the rectified images into desidered dimensions.
    
    After rectification usually the image is no more into the original image bounds.
    One can apply any transformation that do not affect disparity to fit the image into boundaries.
    This function corrects flipped images too.
    The algorithm may fail if one epipole is too close to the image.
    
    Parameters
    ----------
    intrinsicMatrix1, intrinsicMatrix2 : numpy.ndarray
        3x3 original camera matrices of intrinsic parameters.
    H1, H2 : numpy.ndarray
        3x3 rectifying homographies.
    dims1, dims2 : tuple
        Resolution of images as (width, height) tuple.
    distCoeffs1, distCoeffs2 : numpy.ndarray, optional
        Distortion coefficients in the order followed by OpenCV. If None is passed, zero distortion is assumed.
    destDims : tuple, optional
        Resolution of destination images as (width, height) tuple (default to the first image resolution).
    zoom : float, optional
        Zoom parameter to be applied to both images (default to 1). Used to remove unwanted portions of the images.
        
    Returns
    -------
    K1, K2 : numpy.ndarray
        3x3 affine transformations to be used for the first and the second camera, respectively.
        They will differ by a x-shift value only.
    """
    if destDims is None:
        destDims = dims1

    # Get border points
    tL1, tR1, bR1, bL1 = _getCorners(H1, intrinsicMatrix1, dims1, distCoeffs1)
    tL2, tR2, bR2, bL2 = _getCorners(H2, intrinsicMatrix2, dims2, distCoeffs2)
    
    minX1 = min(tR1[0], bR1[0], bL1[0], tL1[0])
    minX2 = min(tR2[0], bR2[0], bL2[0], tL2[0])
    maxX1 = max(tR1[0], bR1[0], bL1[0], tL1[0])
    maxX2 = max(tR2[0], bR2[0], bL2[0], tL2[0])
    
    minY = min(tR2[1], bR2[1], bL2[1], tL2[1], tR1[1], bR1[1], bL1[1], tL1[1])
    maxY = max(tR2[1], bR2[1], bL2[1], tL2[1], tR1[1], bR1[1], bL1[1], tL1[1])
    
    # Flip factor
    flipX = 1
    flipY = 1
    if tL1[0]>tR1[0]:
        flipX = -1
    if tL1[1]>bL1[1]:
        flipY = -1
    
    # Scale X (choose (unique) scale X to best fit bigger image between left and right)
    if(maxX2 - minX2 > maxX1 - minX1):
        scaleX = flipX * zoom * destDims[0]/(maxX2 - minX2)
    else:
        scaleX = flipX * zoom * destDims[0]/(maxX1 - minX1)
    
    # Scale Y (unique not to lose rectification) 
    scaleY = flipY * zoom * destDims[1]/(maxY - minY)
    
    # Translation X (keep always at left border)
    if flipX == 1:
        tX1 = -minX1 * scaleX
        tX2 = -minX2 * scaleX
    else:
        tX1 = -maxX1 * scaleX
        tX2 = -maxX2 * scaleX
    
    # Translation Y (keep always at top border)
    if flipY == 1:
        tY = -minY * scaleY
    else:
        tY = -maxY * scaleY 
    
    # Compensate zoom
    tX1 -= destDims[0]*(zoom-1)/2
    tX2 -= destDims[0]*(zoom-1)/2
    tY -= destDims[1]*(zoom-1)/2
    
    # Final transformations    
    K1 = np.array( [[scaleX,0,tX1], [0,scaleY,tY], [0,0,1]] )
    K2 = np.array( [[scaleX,0,tX2], [0,scaleY,tY], [0,0,1]] )
    
    return K1, K2
    

def _getCorners(H, intrinsicMatrix, dims, distCoeffs=None):
    """
    Get points on the image borders after distortion correction and a rectification transformation.
    
    Parameters
    ----------
    H : numpy.ndarray
        3x3 rectification homography.
    intrinsicMatrix : numpy.ndarray
        3x3 camera matrix of intrinsic parameters.
    dims : tuple
        Image dimensions in pixels as (width, height).
    distCoeffs : numpy.ndarray or None
        Distortion coefficients (default to None).
    
    Returns
    -------
    tuple
        Corners of the image clockwise from top-left.
    """
    if distCoeffs is None:
        distCoeffs = np.zeros(5)
    
    # Set image corners in the form requested by cv2.undistortPoints
    corners = np.zeros((4,1,2), dtype=np.float32)
    corners[0,0] = [0,0]                      # Top left
    corners[1,0] = [dims[0]-1,0]              # Top right
    corners[2,0] = [dims[0]-1,dims[1]-1]      # Bottom right
    corners[3,0] = [0, dims[1]-1]             # Bottom left
    undist_rect_corners = cv2.undistortPoints(corners, intrinsicMatrix, distCoeffs, R=H.dot(intrinsicMatrix))
    
    return [(x,y) for x, y in np.squeeze(undist_rect_corners)]


def _getCornersFromMatrix(M, dims):
    """
    Calculate image corners from homography.
    
    Parameters
    ----------
    M : numpy.ndarray
        3x3 transformation matrix.
    dims : tuple
        Image dimensions as (width, height).
    
    Returns
    -------
    tuple
        The coordinates of the four corners (clockwise from top left).
    """
    tL = M.dot(np.array([[0],[0],[1]]))[:,0]
    tL = tL/tL[2]
    tR = M.dot(np.array([[dims[0]-1],[0],[1]]))[:,0]
    tR = tR/tR[2]
    bR = M.dot(np.array([[dims[0]-1],[dims[1]-1],[1]]))[:,0]
    bR = bR/bR[2]
    bL = M.dot(np.array([[0],[dims[1]-1],[1]]))[:,0]
    bL = bL/bL[2]
    
    return tL[:2], tR[:2], bR[:2], bL[:2]
        


def stereoRectify(rig):
    """
    Rectify the StereoRig object using the standard OpenCV algorithm.
    
    This function computes the new common camera orientation by averaging.
    It does not produce the rectifying homographies with minimal perspective distortion. 
    
    Parameters
    ----------
    rig : StereoRig
        An object of the StereoRig class
        
    Returns
    -------
    rectifiedStereoRigObj : RectifiedStereoRig
        An object of the RectifiedStereoRig class containing the rectifying homographies.
    
    """
    R1, R2, _, _, _, _, _ = cv2.stereoRectify(rig.intrinsic1, rig.distCoeffs1, rig.intrinsic2, rig.distCoeffs2, rig.res1, rig.R, rig.T, flags=0)
    
    # OpenCV does not compute the rectifying homography, but a rotation in the object space.
    # R1 = Rnew * Rcam^{-1}
    # To get the homography:
    homography1 = R1.dot(np.linalg.inv(rig.intrinsic1))
    homography2 = R2.dot(np.linalg.inv(rig.intrinsic2))
    # To get the common orientation, since the first camera has orientation as origin:
    # Rcommon = R1
    # It also can be retrieved from R2, cancelling the rotation of the second camera.
    # Rcommon = R2.dot(np.linalg.inv(rig.R))
    
    rectStereoRig = ss.RectifiedStereoRig(R1, homography1, homography2, rig)
    
    return rectStereoRig



def fusielloRectify(rig):
    """
    Computes the two rectifying homographies and returns a RectifiedStereoRig.
    
    This method uses the algorithm illustrated in *A compact algorithm for rectification of stereo pair*, 
    A. Fusiello et al., Machine Vision and Applications (2000).
    
    Parameters
    ----------
    rig : StereoRig
        An object of the StereoRig class. Camera must be position 1 (origin), projector in position 2.
        
    Returns
    -------
    rectifiedStereoRigObj : RectifiedStereoRig
        An object of the RectifiedStereoRig class containing the rectifying homographies.
    """
    # Get baseline vector
    _, B = rig.getCenters() # First camera is always in origin
    
    # Find new directions
    v1 = np.squeeze(B)                      # New x direction
    v2 = np.cross(np.array([0,0,1]) , v1)   # New y direction. 
    v3 = np.cross(v1,v2)                    # New z direction
    
    # Normalize
    v1 = v1 / np.linalg.norm(v1)    # Normalize x
    v2 = v2 / np.linalg.norm(v2)    # Normalize y
    v3 = v3 / np.linalg.norm(v3)    # Normalize z
    
    
    # Create rotation matrix (new common orientation of the cameras)
    Rot = np.array( [ v1, v2, v3 ] )
    
    # New intrinsic is arbitrary (it needs to be adapted later using fitting matrices).
    A = (rig.intrinsic1 + rig.intrinsic2)/2
    
    # Transformations to rectify images
    Rectify1 = A.dot(Rot).dot( np.linalg.inv(rig.intrinsic1) )
    Rectify2 = A.dot(Rot).dot( np.linalg.inv(rig.R) ).dot(np.linalg.inv(rig.intrinsic2))
    
    rectStereoRig = ss.RectifiedStereoRig(Rot, Rectify1, Rectify2, rig)
    
    return rectStereoRig



def _lowLevelRectify(rig):
    """
    Get basic rectification using Fusiello et al.
    for *internal* purposes only.
    
    This assumes that camera is coincident with world origin.
    Please refer to the rectification module for general
    image rectification.
    
    See Also
    --------
    :func:`simplestereo.rectification.fusielloRectify`
    """
    
    # Get baseline vector
    _, B = rig.getCenters()
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
    R1 = ( R ).dot( np.linalg.inv(rig.intrinsic1) )
    R2 = ( R ).dot( np.linalg.inv(rig.R) ).dot( np.linalg.inv(rig.intrinsic2) )
    
    return R1, R2, R


def loopRectify(rig):
    """
    Computes the two rectifying homographies and returns a RectifiedStereoRig.
    
    This method is an implementation of the algorithm illustrated in 
    *Computying rectifying homographies for stereo vision*, CVPR 1999, Loop C. and Zhang Z.
    This function performs a minimization using ``scipy.optimize`` module.
    
    Parameters
    ----------
    rig : StereoRig
        An object of the StereoRig class.
        
    Returns
    -------
    rectifiedStereoRigObj : RectifiedStereoRig
        An object of the RectifiedStereoRig class containing the rectifying homographies.
    
    
    .. note:: 
       Also an object of :meth:`simplestereo.RectifiedStereoRig` may be
       passed as input to recalculate its rectification
       transformations (e.g. changing algorithm).
    """
    
    def findInitialGuess(A1, B1, A2, B2): # Internal use function
        # Find initial guess for optimization
        try:
            D1 = cholesky(A1, lower=True) # Upper triangle so that A1 = D1.T.dot(D1)
            D2 = cholesky(A2, lower=True)
        except:
            # If factorization fails because of negative eigenvalues
            # you may try to manage with it...
            # Eg. try to add a small value to diagonal elements
            # BUT THIS IS NOT GUARANTEED
            A1 += 1e-10 * np.eye(3) 
            A2 += 1e-10 * np.eye(3)
            try:
                D1 = cholesky(A1, lower=True)
                D2 = cholesky(A2, lower=True)
                warnings.warn("Added 1e-10 value to diagonal elements of A1 and A2 before Cholesky factorization.", RuntimeWarning)
            except np.linalg.LinAlgError:
                # If fails again, raise the original error
                raise e
                
        # Calculate the eigenvector associated to the maximum eigenvalue of np.linalg.inv(D1).T.dot(B1).dot(np.linalg.inv(D1))
        D1_inv = np.linalg.inv(D1)
        eval1, evec1 = np.linalg.eig(D1_inv.T.dot(B1).dot(D1_inv))   # Calculate corresponding eigenvectors/values
        evec1_max = evec1[ np.argmax(eval1) ]                        # Take eigenvector associated to greates eigenvalue
        z1 = D1_inv.dot(evec1_max)                                   # Initial guess associated with first image
        
        # Same for image 2
        D2_inv = np.linalg.inv(D2)
        eval2, evec2 = np.linalg.eig(D2_inv.T.dot(B2).dot(D2_inv))
        evec2_max = evec2[ np.argmax(eval2) ]
        z2 = D2_inv.dot(evec2_max)                             
         
        # Initial guess chosen as the average
        z = ( z1/np.linalg.norm(z1) + z2/np.linalg.norm(z2) ) / 2
        
        return z


    def minDistortion(z, A1, B1, A2, B2): 
        # Distortion minimization target
        z[1] = 1 # Impose z in the form [lambda, 1, 0]
        z[2] = 0
        return float( z.T.dot(A1).dot(z) / z.T.dot(B1).dot(z) + z.T.dot(A2).dot(z) / z.T.dot(B2).dot(z) )


    def getMinYCoord(H, dims):
        # Get the minimum Y coordinate after a transformation H.
        # Please refer to "Computying rectifying homographies for stereo vision", CVPR 1999, Loop C. and Zhang Z.
        tL = H.dot(np.array([[0],[0],[1]]))[:,0]
        tL = tL/tL[2]
        bL = H.dot(np.array([[0],[dims[1]-1],[1]]))[:,0]
        bL = bL/bL[2]
        tR = H.dot(np.array([[dims[0]-1],[0],[1]]))[:,0]
        tR = tR/tR[2]
        bR = H.dot(np.array([[dims[0]-1],[dims[1]-1],[1]]))[:,0]
        bR = bR/bR[2]
        
        return min(tL[1], tR[1], bR[1], bL[1])
    
    # Get data from stereo rig
    F = rig.getFundamentalMatrix()
    dims1 = rig.res1
    dims2 = rig.res2
    
    # Calculate epipoles as left and right kernels of F
    e1 = null_space(F)
    #e2 = null_space(F.T) # Not needed, but kept for reference
    
    # Get e1 as cross product antisymmetric matrix
    e1_cross = ss.utils.getCrossProductMatrix(np.squeeze(e1))
    
    # Compute A and B matrices for both images
    A1 = e1_cross.T.dot( (dims1[0]*dims1[1]/12)*np.array([[dims1[0]**2 - 1, 0, 0],[0, dims1[1]**2 - 1,0],[0, 0, 0]]) ).dot(e1_cross)
    A2 = F.T.dot( (dims2[0]*dims2[1]/12)*np.array([[dims2[0]**2 - 1, 0, 0],[0, dims2[1]**2 - 1,0],[0, 0, 0]]) ).dot(F)
    B1 = e1_cross.T.dot( np.array([[(dims1[0] - 1)**2/4, (dims1[0] - 1)*(dims1[1] - 1)/4, (dims1[0] - 1)/2], [(dims1[0] - 1)*(dims1[1] - 1)/4, (dims1[1] - 1)**2/4, (dims1[1] - 1)/2],[(dims1[0] - 1)/2, (dims1[1] - 1)/2, 1]]) ).dot(e1_cross)
    B2 = F.T.dot( np.array([[(dims2[0] - 1)**2/4, (dims2[0] - 1)*(dims2[1] - 1)/4, (dims2[0] - 1)/2], [(dims2[0] - 1)*(dims2[1] - 1)/4, (dims2[1] - 1)**2/4, (dims2[1] - 1)/2],[(dims2[0] - 1)/2, (dims2[1] - 1)/2, 1]]) ).dot(F)
    
    # Find initial guess (see par. 5.2 of paper)
    initial_guess = findInitialGuess(A1, B1, A2, B2)
    
    # Minimize using default method
    result = optimize.minimize(minDistortion, initial_guess, args=(A1, B1, A2, B2,))
    if result.success:
        z = result.x
    else:
        raise ValueError(result.message)
    
    # Impose z in the form [lambda, 1, 0]
    z[1] = 1
    z[2] = 0
    
    # Get w1 and w2
    w1 = e1_cross.dot(z)
    w2 = F.dot(z)
    w1 = w1/w1[2]
    w2 = w2/w2[2]
    
    # Build projective transforms
    Hp1 = np.array([ [1,0,0], [0,1,0], w1])
    Hp2 = np.array([ [1,0,0], [0,1,0], w2])
    
    # Calculate vc2 so that "the minimum w-coordinate of a pixel in either image is zero."
    vc2 = -min( getMinYCoord(Hp1, dims1), getMinYCoord(Hp2, dims2) )
    
    # Build similarity transforms
    
    # Original formulation (NOT WORKING)
    #Hr1 = np.array([ [ F[2,1]-w1[1]*F[2,2], w1[0]*F[2,2]-F[2,0], 0], \
    #                 [ F[2,0]-w1[0]*F[2,2], F[2,1]-w1[1]*F[2,2], F[2,2] + vc2 ], \
    #                 [0, 0, 1] ])
    
    # Changed sign to second row of Hr1 to make it work...
    Hr1 = np.array([ [ F[2,1]-w1[1]*F[2,2], w1[0]*F[2,2]-F[2,0], 0], \
                     [ w1[0]*F[2,2]-F[2,0], w1[1]*F[2,2]-F[2,1], -(F[2,2] + vc2) ], \
                     [0, 0, 1] ]) 
    
    # This one is like the original
    Hr2 = np.array([ [ F[1,2]-w2[1]*F[2,2], w2[0]*F[2,2]-F[0,2], 0], \
                     [ F[0,2]-w2[0]*F[2,2], F[1,2]-w2[1]*F[2,2], vc2], \
                     [0, 0, 1] ])
    
    # Combine perspective and similarity transforms
    Hrp1 = Hr1.dot(Hp1)
    Hrp2 = Hr2.dot(Hp2)
    
    # Find shearing transforms
    Hs1 = getBestXShearingTransformation(Hrp1, dims1)
    Hs2 = getBestXShearingTransformation(Hrp2, dims2)
    
    # Get final rectification transforms
    Rectify1 = Hs1.dot(Hrp1)
    Rectify2 = Hs2.dot(Hrp2)
    
    # END OF ORIGINAL ALGORITHM
    
    ### Rcommon to be calculated here!
    # New x axis
    C1 , C2 = rig.getCenters()
    xv = C1 - C2                    # New x axis
    
    # Calculation of the z axis of common orientation (thanks to Marta)
    zv = np.cross(e1[:,0],z)        # New z axis
    zv = zv/zv[2]
    
    # Get y axis as cross product
    yv = np.cross(zv, xv)           # New y axis
    
    xv = xv / np.linalg.norm(xv)    # Normalize x direction
    yv = yv / np.linalg.norm(yv)    # Normalize y direction
    zv = zv / np.linalg.norm(zv)    # Normalize z direction
    
    # Build common camera orientation
    Rcommon = np.array([xv,yv,zv])
    
    rectStereoRig = ss.RectifiedStereoRig(Rcommon, Rectify1, Rectify2, rig)
    
    return rectStereoRig



def getBestXShearingTransformation(rectHomography, dims):
    """
    Get best shear transformation (affine) over x axis that minimizes distortion.
    
    Applying a shearing transformation over the x axis does not affect rectification and allows to reduce
    image distortion.
    Original algorithm in par. 7 of *Computying rectifying homographies for stereo vision*, CVPR 1999, Loop C. and Zhang Z.
    
    Parameters
    ----------
    rectHomography : numpy.ndarray
        A 3x3 rectification homography.
    dims : tuple
        Resolution of destination image as (width, height) tuple.
    
    Returns
    -------
    S : numpy.ndarray
        A 3x3 shearing (x axis) transform.
    
    
    .. note::
       All the tranformations applied to the images must be taken into account when computing disparity
       for 3D reconstruction.
    """
    a = rectHomography.dot([(dims[0]-1)/2, 0, 1]) # Top middlepoint
    b = rectHomography.dot([(dims[0]-1), (dims[1]-1)/2, 1]) # Right middlepoint
    c = rectHomography.dot([(dims[0]-1)/2, (dims[1]-1), 1]) # Bottom middlepoint
    d = rectHomography.dot([0, (dims[1]-1)/2, 1]) # Left middlepoint
    a = a / a[2]
    b = b / b[2]
    c = c / c[2]
    d = d / d[2]
    
    # Get lines
    x = b - d
    y = c - a
    
    # Calculate coefficients
    a_coeff = ( (dims[1]*x[1])**2 + (dims[0]*y[1])**2 ) / ( dims[0]*dims[1]*(x[1]*y[0] - x[0]*y[1]) )
    b_coeff = ( (dims[1]**2)*x[0]*x[1] + (dims[0]**2)*y[0]*y[1] ) / ( dims[0]*dims[1]*(x[0]*y[1] - x[1]*y[0]) )
    
    # Build shearing matrix transform
    S = np.array([[a_coeff,b_coeff,0],[0,1,0],[0,0,1]])
    
    return S



def directRectify(rig):
    """
    Compute the analytical rectification homographies.
    
    Compute the 3x3 transformations to rectify a couple of stereo images
    with minimim perspective distortion.
    This implementation provides direct analytic solution, without using
    minimization.
    
    Parameters
    ----------
    rig : StereoRig
        An object of the StereoRig class.
    
    Returns
    -------
    Rectify1, Rectify2 : numpy.ndarray
        3x3 rectification homographies.
        
        
    See Also
    --------
    Lafiosca Pasquale and Ceccaroni Marta, "Rectifying homographies for 
    stereo vision: analytical solution for minimal distortion", Lecture 
    Notes in Networks and Systems, 2022.
    """
    # Load data from stereo rig
    A1 = rig.intrinsic1
    A2 = rig.intrinsic2
    RT1 = np.hstack((np.eye(3), np.zeros((3,1))))   # World origin set in first camera
    RT2 = np.hstack((rig.R, rig.T))
    dims1 = rig.res1
    dims2 = rig.res2
    F = rig.getFundamentalMatrix()
    
    def getMinYCoord(H, dims):
        # Get the minimum Y coordinate after a transformation H.
        # Please refer to "Computying rectifying homographies for stereo vision", CVPR 1999, Loop C. and Zhang Z.
        tL = H.dot(np.array([[0],[0],[1]]))[:,0]
        tL = tL/tL[2]
        bL = H.dot(np.array([[0],[dims[1]-1],[1]]))[:,0]
        bL = bL/bL[2]
        tR = H.dot(np.array([[dims[0]-1],[0],[1]]))[:,0]
        tR = tR/tR[2]
        bR = H.dot(np.array([[dims[0]-1],[dims[1]-1],[1]]))[:,0]
        bR = bR/bR[2]
        return min(tL[1], tR[1], bR[1], bL[1])
        
    if np.all(np.equal(F/F[2,1], np.array([[0,0,0],[0,0,-1],[0,1,0]]))):
        # PARTICULAR CASE 1: Stereo rig is already rectified
        # No perspective transformation is needed
        w1 = w2 = np.array([0,0,1])
    
    else:
        # Baseline vector in world coord (cam1 -> cam2)
        bv = np.linalg.inv(RT2[:,:3]).dot(RT2[:,3]) - np.linalg.inv(RT1[:,:3]).dot(RT1[:,3])
        
        # Auxiliary matrices
        B = ( bv.dot(bv) * np.eye(3) - bv[:,None].dot(bv[None,:]) ).dot(np.linalg.inv(A1.dot(RT1[:,:3])))
        L1 = np.transpose(np.linalg.inv(A1.dot(RT1[:,:3]))).dot(B)
        L2 = np.transpose(np.linalg.inv(A2.dot(RT2[:,:3]))).dot(B)
        
        # Auxiliary matrices II (as in Loop-Zhang algorithm)
        # N.B. The variable P1 is actually P1.P1^T and so on.
        P1 = (dims1[0]*dims1[1]/12)*np.array([[dims1[0]**2 - 1, 0, 0],[0, dims1[1]**2 - 1,0],[0, 0, 0]])
        Pc1 = np.array([[(dims1[0] - 1)**2/4, (dims1[0] - 1)*(dims1[1] - 1)/4, (dims1[0] - 1)/2], [(dims1[0] - 1)*(dims1[1] - 1)/4, (dims1[1] - 1)**2/4, (dims1[1] - 1)/2],[(dims1[0] - 1)/2, (dims1[1] - 1)/2, 1]])
        P2 = (dims2[0]*dims2[1]/12)*np.array([[dims2[0]**2 - 1, 0, 0],[0, dims2[1]**2 - 1,0],[0, 0, 0]])
        Pc2 = np.array([[(dims2[0] - 1)**2/4, (dims2[0] - 1)*(dims2[1] - 1)/4, (dims2[0] - 1)/2], [(dims2[0] - 1)*(dims2[1] - 1)/4, (dims2[1] - 1)**2/4, (dims2[1] - 1)/2],[(dims2[0] - 1)/2, (dims2[1] - 1)/2, 1]])
        
        M1 = L1.T.dot(P1).dot(L1)
        C1 = L1.T.dot(Pc1).dot(L1)
        M2 = L2.T.dot(P2).dot(L2)
        C2 = L2.T.dot(Pc2).dot(L2)
        
        # Polynomial coefficients
        m1 = M1[1,2]*C1[1,2] - M1[2,2]*C1[1,1]
        m2 = M1[1,1]*C1[1,2] - M1[1,2]*C1[1,1]
        
        if np.all(np.equal(RT1[:,:3], RT2[:,:3])) and np.all(np.equal(A1, A2)) and np.all(np.equal(P1, P2)) and np.all(np.equal(Pc1, Pc2)):
            # PARTICULAR CASE 2: The cameras have the same orientation: we have a single solution
            sol = [-m1/m2]
            
        else:
            # Polynomial coefficients II
            m3 = C2[1,2]/C2[1,1]
            m4 = C2[1,1]/C1[1,1]
            m5 = M2[1,2]*C2[1,2] - M2[2,2]*C2[1,1]
            m6 = M2[1,1]*C2[1,2] - M2[1,2]*C2[1,1]
            m7 = C1[1,2]/C1[1,1]
            m8 = 1/m4
            
            a = m2*m4 + m6*m8
            b = m1*m4 + 3*m2*m3*m4 + m5*m8 + 3*m6*m7*m8
            c = 3*(m1*m3*m4 + m2*m3**2*m4 + m5*m7*m8 + m6*m7**2*m8)
            d = 3*m1*m3**2*m4 + m2*m3**3*m4 + 3*m5*m7**2*m8 + m6*m7**3*m8
            e = m1*m3**3*m4 + m5*m7**3*m8
            
            # 4th degree equation formula
            p = (8*a*c - 3 * b**2 ) / (8 * a**2)
            q = 12*a*e - 3*b*d + c**2
            s = 27*a*d**2 - 72*a*c*e + 27*b**2*e - 9*b*c*d + 2*c**3
            D0 = math.pow( (1/2)*(s+math.sqrt(s**2 - 4*q**3)), 1/3)
            Q = (1/2) * math.sqrt( -(2/3)*p + 1/(3*a) * (D0 + q / D0) )
            S = ( 8*a**2*d - 4*a*b*c + b**3 ) / ( 8*a**3 ) 
            
            # Take acceptable solutions only
            sol = []
            if -4*Q**2 - 2*p + S/Q >= 0:
                sol.append( -b / (4*a) - Q - (1/2)*math.sqrt( -4*Q**2 - 2*p + S/Q) )
                sol.append( -b / (4*a) - Q + (1/2)*math.sqrt( -4*Q**2 - 2*p + S/Q) )
            
            if -4*Q**2 - 2*p - S/Q >= 0:
                sol.append( -b / (4*a) + Q - (1/2)*math.sqrt( -4*Q**2 - 2*p - S/Q) )
                sol.append( -b / (4*a) + Q + (1/2)*math.sqrt( -4*Q**2 - 2*p - S/Q) )
            
            if len(sol)<1:
                raise ValueError("No analitic solution.")
        
           
        def evaluateSolution(ss):
            # Inner function to compute w1 and w2 from the solution
            
            # Point over image 1 in world coordinates
            p1w = np.linalg.inv(RT1[:,:3]).dot( np.linalg.inv(A1).dot(np.array([0,ss,1])) - RT1[:,3] )
            # New x axis
            xv = bv / np.linalg.norm(bv)
            # Projection on the baseline of the vector p1w - C2 in world coordinates
            oop1w = ( p1w + np.linalg.inv(RT2[:,:3]).dot(RT2[:,3]) ).dot(xv) * xv - np.linalg.inv(RT2[:,:3]).dot(RT2[:,3])
            
            zv = p1w - oop1w                # New z axis
            yv = np.cross(zv, bv)           # New y axis
            yv = yv / np.linalg.norm(yv)    # Normalize y direction
            zv = zv / np.linalg.norm(zv)    # Normalize z direction
            Rnew = np.array([xv,yv,zv])     # New camera orientation
            
            # Loop-Zhang w1 and w2
            w1 = Rnew.dot( np.linalg.inv(A1.dot(RT1[:,:3])) )[2,:]
            w2 = Rnew.dot( np.linalg.inv(A2.dot(RT2[:,:3])) )[2,:]
            w1 = w1 / w1[2]                 # Rescale with 3rd coordinate as 1
            w2 = w2 / w2[2]
            #l = -w1[1]/w1[0]               # Loop-Zhang lambda parameter (not needed)
            
            return w1, w2, Rnew
        
        
        def getDistortion(s):
            # Inner function as compact version of Loop and Zhang distortion
            w1, w2, _ = evaluateSolution(s)    
            dist1 = float( w1.dot(P1).dot(w1)/w1.dot(Pc1).dot(w1) )
            dist2 = float( w2.dot(P2).dot(w2)/w2.dot(Pc2).dot(w2) )
            return dist1+dist2
            
        
        # Find minimum distortion among admissible solutions (4 or 2 solutions)
        bestSol = min(zip( sol, map(getDistortion, sol)), key=lambda x:x[1])[0]
        # Get associated w1, w2 and new common orientation.
        w1, w2, Rnew = evaluateSolution(bestSol)
    
    # At this point we have the correct w1 and w2
    # From here we follow the rest of the Loop-Zhang algorithm
        
    # Build projective transforms
    Hp1 = np.array([ [1,0,0], [0,1,0], w1 ])
    Hp2 = np.array([ [1,0,0], [0,1,0], w2 ])
    
    # Calculate vc2 so that "the minimum w-coordinate of a pixel in either image is zero."
    vc2 = -min( getMinYCoord(Hp1, dims1), getMinYCoord(Hp2, dims2) )
    
    # Build similarity transforms
    Hr1 = np.array([ [F[2,1]-w1[1]*F[2,2], w1[0]*F[2,2]-F[2,0], 0], \
                     [w1[0]*F[2,2]-F[2,0], w1[1]*F[2,2]-F[2,1], -(F[2,2] + vc2)], \
                     [0, 0, 1] ]) 
    
    Hr2 = np.array([ [F[1,2]-w2[1]*F[2,2], w2[0]*F[2,2]-F[0,2], 0], \
                     [F[0,2]-w2[0]*F[2,2], F[1,2]-w2[1]*F[2,2], vc2], \
                     [0, 0, 1] ])
    
    # Combine perspective and similarity transformations
    Hrp1 = Hr1.dot(Hp1)
    Hrp2 = Hr2.dot(Hp2)
    
    # Find best shearing transformations
    Hs1 = getBestXShearingTransformation(Hrp1, dims1)
    Hs2 = getBestXShearingTransformation(Hrp2, dims2)
    
    # Get final rectification transformations
    Rectify1 = Hs1.dot(Hrp1)
    Rectify2 = Hs2.dot(Hrp2)
    
    # Build a RectifiedStereoRig object
    rectStereoRig = ss.RectifiedStereoRig(Rnew, Rectify1, Rectify2, rig)
    
    return rectStereoRig
