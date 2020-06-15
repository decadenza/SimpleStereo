'''
postprocessing
==============
Functions to manage disparity maps and point clouds.
'''
import numpy as np

def exportPoints(points3D, filepath):
    """
    Export raw point cloud to PLY file.
    
    Parameters
    ----------
    points3D : numpy.ndarray
        3D points with shape (height, width, 3) where last dimension contains ordered x,y,z coordinates.
    filepath : str
        File path for the PLY file (absolute or relative).
    
    Notes
    -----
    *points3D* should be calculated from cv2.reprojectImageTo3D(disparityMap, Q).
    """
    
    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\ncomment SimpleStereo point cloud export\n")
        f.write("element vertex {}\n".format(points3D.shape[0]*points3D.shape[1]))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x,y,z in points3D.reshape(points3D.shape[0]*points3D.shape[1],3):
            f.write("{} {} {}\n".format(x, -y, z))
    
    
