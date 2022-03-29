import sys
import os

import numpy as np
import cv2

import simplestereo as ss


if __name__ == "__main__":
    
    CURPATH = os.path.dirname(os.path.realpath(__file__))
    
    # Period of fringe image (pixels)
    period = 8
    
    # Images path
    IMG_ORIGINAL_FRINGE = os.path.join(CURPATH, "res", "stereoFTP", "fringe8.png")
    IMG_OBJECT = os.path.join(CURPATH, "res", "stereoFTP", "ellipsoid8.png")
    # Output
    SAVEPATH = os.path.join(CURPATH, "res", "stereoFTP", "stereoFTP_output.ply")
    
    # Load stereo rig
    stereoRig = ss.StereoRig.fromFile(os.path.join(CURPATH,"res", "stereoFTP","stereoRig.json"))
    
    # Load fringe image
    imgFringe = cv2.imread(IMG_ORIGINAL_FRINGE)  # Original projected fringe pattern
    
    ### Initialize FTP manager
    FTPManager = ss.active.StereoFTP(stereoRig, imgFringe, period)
    
    # Custom unwrapping method (OPTIONAL)
    chosenUnwrappingMethod = None
    #chosenUnwrappingMethod = lambda phase : ss.unwrapping.infiniteImpulseResponse(phase, tau=0.8)
    
    
    print(f"StereoFTP initialized.\nperiod = {period}")
    print(f"Object = {IMG_OBJECT}")
        
    ### Intialization END
    
    # Retrieve object image
    imgObj = cv2.imread(IMG_OBJECT)                  
    
    # Image ROI (optional)
    roi = ss.active.computeROI(imgObj, blackThreshold=10, extraMargin=150)
    roi_x, roi_y, roi_w, roi_h = roi # Needed for export only
    
    '''
    # Intereactively select on image
    roi = cv2.selectROI("Select ROI", imgObj)
    roi_x, roi_y, roi_w, roi_h = roi
    cv2.destroyWindow("Select ROI")
    '''
    
    '''
    # Or define roi MANUALLY
    # Care must be taken to remove non illuminated areas
    roi_x = 911
    roi_y = 334
    roi_w = 1280 - roi_x
    roi_h = 701 - roi_y
    roi = (roi_x, roi_y, roi_w, roi_h)
    ''' 
    
    
    print("ROI:", roi)
    
    # Scan
    finalPoints = FTPManager.getCloud(imgObj, roi=roi, unwrappingMethod=chosenUnwrappingMethod, plot=True)
    
    # Export in subdir
    ss.points.exportPLY(finalPoints, SAVEPATH, 
        referenceImage=imgObj[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w])
    
    print(f"Saved in {SAVEPATH}")
    
