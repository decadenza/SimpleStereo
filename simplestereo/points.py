'''
postprocessing
==============
Functions to manage disparity maps and point clouds.
'''
import numpy as np
import cv2

def exportPLY(points3D, filepath, referenceImage=None, precision=6):
    """
    Export raw point cloud to PLY file (ASCII).
    
    Parameters
    ----------
    points3D : numpy.ndarray
        Array of 3D points. The last dimension must contain ordered x,y,z coordinates.
    filepath : str
        File path for the PLY file (absolute or relative).
    referenceImage : numpy.ndarray, optional
        Reference image to extract color from. It must contain the same
        number of points of `points3D`. Last dimension must be either
        1 (grayscale) or 3 (BGR).
        Default to None.
    precision : int
        Decimal places to save coordinates with. Higher precision causes
        bigger file size.
        Default to 6.
    """
    
    points3D = points3D.reshape(-1,3)
    n = points3D.shape[0]
    
    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\ncomment SimpleStereo point cloud export\n")
        f.write("element vertex {}\n".format(n))
        f.write("property double x\nproperty double y\nproperty double z\n")
        
        if referenceImage is None:
            f.write("end_header\n")
            for x,y,z in points3D:
                f.write("{:.6f} {:.6f} {:.6f}\n".format(x, y, z))
        else:
            if referenceImage.size == points3D.size:
                # Assuming BGR image (OpenCV compatible) (3 color values)
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("end_header\n")
                referenceImage = referenceImage.reshape(-1,3)
                for i in range(n):
                    # Precision limited to p decimal places.
                    f.write("{:.{p}f} {:.{p}f} {:.{p}f} {:d} {:d} {:d}\n".format(
                        points3D[i,0], points3D[i,1], points3D[i,2], 
                        referenceImage[i,2], referenceImage[i,1], referenceImage[i,0], p=precision)) # Invert BGR to RGB
            else:
                # Assuming grayscale image (1 color value)
                f.write("property uchar intensity\n")
                f.write("end_header\n")
                referenceImage = np.ravel(referenceImage)
                for i in range(n):
                    # Precision limited to p decimal places.
                    f.write("{:.{p}f} {:.{p}f} {:.{p}f} {:d}\n".format(
                        points3D[i,0], points3D[i,1], points3D[i,2], 
                        referenceImage[i], p=precision)) # Grayscale
    

def importPLY(filename, *properties):
    """
    Import 3D coordinates from PLY file.
    
    Parameters
    ----------
    filename : str
        PLY file path.
    *properties : argument list, optional
        Property column positions to be extracted as `float`, in the
        same order. Default to (0,1,2).
        
    
    Returns
    -------
    numpy.ndarray
        Array of data values with shape (number of values, number of
        properties).
        
    ..todo::
        Automatically read PLY properties as `dict`.
        Manage values other than `float`.
    """
    if not properties:
        properties = (0,1,2)
    
    with open(filename, "r") as f:
        i=0
        for line in f:
            i+=1
            if line.rstrip().lower() == "end_header":
                break
        points = []
        for line in f:
            prop = line.split(' ')
            points.append([ float(prop[x]) for x in properties ])
        
    return np.asarray(points, dtype=float)



def getAdimensional3DPoints(disparityMap):
    """
    Get adimensional 3D points from the disparity map.
    
    This is the adimensional version of
    `RectifiedStereoRig.get3DPoints()`.
    Useful to reconstruct non-metric 3D models from any disparity map
    when the stereo rig object is not known.
    It may lead to incorrect proportions.
    
    Parameters
    ----------
    disparityMap : numpy.ndarray
        A dense disparity map having same height and width of images.
    
    Returns
    -------
    numpy.ndarray
        Array of points having shape *(height,width,3)*, where at each y,x coordinates
        a *(x,y,z)* point is associated.
    
    See Also
    --------
    :meth:`RectifiedStereoRig.get3DPoints`
    """
    height, width = disparityMap.shape[:2]
    
    b   = 1
    fx  = width
    fy  = width
    cx1 = width/2
    cx2 = width/2
    a1  = 0
    a2  = 0
    cy  = height/2
    
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
