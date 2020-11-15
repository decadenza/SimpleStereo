'''
postprocessing
==============
Functions to manage disparity maps and point clouds.
'''
import numpy as np

def exportPoints(points3D, filepath, referenceImage=None):
    """
    Export raw point cloud to PLY file.
    
    Assuming camera z-axis to be horizontal (parallel to ground), old z-axis
    is mapped to new y-axis for easy visualization on point cloud softwares.
    X-axis remains the same, and new z-axis is oriented as standard cartesian
    coordinate system.
    
    Parameters
    ----------
    points3D : numpy.ndarray
        3D points with shape (height, width, 3) where last dimension contains ordered x,y,z coordinates.
    filepath : str
        File path for the PLY file (absolute or relative).
    referenceImage : numpy.ndarray
        BGR reference image (used for disparity map) to extract color from
        having the same width and height. Default to None.
    
    Notes
    -----
    *points3D* should be calculated from cv2.reprojectImageTo3D(disparityMap, Q).
    """
    
    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\ncomment SimpleStereo point cloud export\n")
        f.write("element vertex {}\n".format(points3D.shape[0]*points3D.shape[1]))
        f.write("property float x\nproperty float y\nproperty float z\n")
        
        if referenceImage is None:
            f.write("end_header\n")
            for x,y,z in points3D.reshape(points3D.shape[0]*points3D.shape[1],3):
                # Precision limited to 6 decimal places.
                f.write("{:.6f} {:.6f} {:.6f}\n".format(x, y, z))
        else:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            if referenceImage.ndim > 2:
                # BGR
                for i in range(points3D.shape[0]):  # height
                    for j in range(points3D.shape[1]): # width
                        # Precision limited to 6 decimal places.
                        f.write("{:.6f} {:.6f} {:.6f} {:d} {:d} {:d}\n".format(
                            points3D[i,j,0], points3D[i,j,1], points3D[i,j,2], 
                            referenceImage[i,j,2], referenceImage[i,j,1], referenceImage[i,j,0])) # Invert BGR to RGB
            else:
                # GRAYSCALE
                for i in range(points3D.shape[0]):  # height
                    for j in range(points3D.shape[1]): # width
                        # Precision limited to 6 decimal places.
                        f.write("{:.6f} {:.6f} {:.6f} {:d} {:d} {:d}\n".format(
                            points3D[i,j,0], points3D[i,j,1], points3D[i,j,2], 
                            referenceImage[i,j], referenceImage[i,j], referenceImage[i,j])) # Grayscale
    
