'''
postprocessing
==============
Functions to manage disparity maps and point clouds.
'''
import numpy as np

def exportPoints(points3D, filepath, referenceImage=None, precision=6):
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
    precision : int
        Decimal places to save coordinates with. Higher precision causes
        bigger file size.
        Default to 6.
    Notes
    -----
    *points3D* should be calculated from cv2.reprojectImageTo3D(disparityMap, Q).
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
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            if referenceImage.ndim == 3:
                # BGR
                referenceImage = referenceImage.reshape(-1,3)
                for i in range(n):
                    # Precision limited to p decimal places.
                    f.write("{:.{p}f} {:.{p}f} {:.{p}f} {:d} {:d} {:d}\n".format(
                        points3D[i,0], points3D[i,1], points3D[i,2], 
                        referenceImage[i,2], referenceImage[i,1], referenceImage[i,0], p=precision)) # Invert BGR to RGB
            else:
                # GRAYSCALE
                referenceImage = referenceImage.reshape(-1,1)
                for i in range(n):
                    # Precision limited to p decimal places.
                    f.write("{:.{p}f} {:.{p}f} {:.{p}f} {:d} {:d} {:d}\n".format(
                        points3D[i,0], points3D[i,1], points3D[i,2], 
                        referenceImage[i], referenceImage[i], referenceImage[i], p=precision)) # Grayscale
    

def importPoints(filename, x=0, y=1, z=2):
    """
    Import 3D coordinates from PLY file.
    
    Parameters
    ----------
    filename : str
        PLY file path.
    x
    y
    z : int, optional
        Coordinate position on each data line.
        Default x y z coordinates expected at the beginning of line.
    
    Returns
    -------
    numpy.ndarray
        Array of 3D xyz points.
        
    ..todo::
        Automatically read PLY properties and load accordingly.
    """
    with open(filename, "r") as f:
        i=0
        for line in f:
            i+=1
            if line.rstrip().lower() == "end_header":
                break
        points = []
        for line in f:
            prop = line.split(' ')
            points.append([ float(prop[x]), float(prop[y]), float(prop[z]) ])
        
    return np.asarray(points, dtype=float)
