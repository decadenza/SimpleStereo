'''
postprocessing
==============
Functions to manage disparity maps and point clouds.
'''
import numpy as np

def exportPoints(points3D, filepath, referenceImage=None):
    """
    Export raw point cloud to PLY file.
    
    Parameters
    ----------
    points3D : numpy.ndarray
        Array of 3D points. The last dimension must contain ordered x,y,z coordinates.
    filepath : str
        File path for the PLY file (absolute or relative).
    referenceImage : numpy.ndarray
        Reference image to extract color from. It must contain the same
        number of points of `points3D`. Last dimension must be either
        1 (grayscale) or 3 (BGR).
        Default to None.
    
    Notes
    -----
    *points3D* should be calculated from cv2.reprojectImageTo3D(disparityMap, Q).
    """
    
    points3D = points3D.reshape(-1,3)
    n = points3D.shape[0]
    
    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\ncomment SimpleStereo point cloud export\n")
        f.write("element vertex {}\n".format(n))
        f.write("property float x\nproperty float y\nproperty float z\n")
        
        if referenceImage is None:
            f.write("end_header\n")
            for x,y,z in points3D:
                # Precision limited to 6 decimal places.
                f.write("{:.6f} {:.6f} {:.6f}\n".format(x, y, z))
        else:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            if referenceImage.ndim == 3:
                # BGR
                referenceImage = referenceImage.reshape(-1,3)
                for i in range(n):
                    # Precision limited to 6 decimal places.
                    f.write("{:.6f} {:.6f} {:.6f} {:d} {:d} {:d}\n".format(
                        points3D[i,0], points3D[i,1], points3D[i,2], 
                        referenceImage[i,2], referenceImage[i,1], referenceImage[i,0])) # Invert BGR to RGB
            else:
                # GRAYSCALE
                referenceImage = referenceImage.reshape(-1,1)
                for i in range(n):
                    # Precision limited to 6 decimal places.
                    f.write("{:.6f} {:.6f} {:.6f} {:d} {:d} {:d}\n".format(
                        points3D[i,0], points3D[i,1], points3D[i,2], 
                        referenceImage[i], referenceImage[i], referenceImage[i])) # Grayscale
    
