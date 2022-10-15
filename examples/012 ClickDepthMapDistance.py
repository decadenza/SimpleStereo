"""
Creating a depth map that can be clicked on using the computer mouse to extract distance estimate

"""

import sys
import os

import numpy as np
import cv2
from scipy.ndimage import median_filter 
from scipy.signal.signaltools import wiener
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize

import simplestereo as ss


#set mouseclick event
def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print (x,y,disp[y,x],filteredImg[y,x])
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        print('Distance: '+ str(Distance)+' m')

# Mouseclick callback
wb=Workbook()
ws=wb.active


### Call OpenCV passive stereo algorithms...
# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
# NB Final disparity will be multiplied by 16 internally! Divide by 16 to get real value.
stereo = cv2.StereoSGBM_create(minDisparity=20, numDisparities=80, blockSize=11, uniquenessRatio=0,P1=50,P2=20)
#disparityMap = stereo.compute(img1_rect, img2_rect).astype(np.float32)/16 # disparityMap coming from Stereo_SGBM is multiplied by 16

# ALTERNATIVE
# Call other SimpleStereo algorithms (much slower)
#stereo = ss.passive.StereoASW(winSize=35, minDisparity=40, maxDisparity=90, gammaC=20, gammaP=17.5, consistent=False)
#stereo = ss.passive.StereoASW(winSize=35, minDisparity=10, maxDisparity=30, gammaC=20, gammaP=17.5, consistent=False)
# Get disparity map
# Returned disparity is unsigned int 16 bit.
#disparityMap = stereo.compute(img1_rect, img2_rect)

#############################################

# Filtering
kernel= np.ones((3,3),np.uint8)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


while True:

    # Read right and left image (please mantain the order as it was in calibration!!!)
    img1 = cv2.imread('res/2/lawn_L.png')  # L
    img2 = cv2.imread('res/2/lawn_R.png')  # R

    # Load rectified stereo rig from file
    rigRect = ss.RectifiedStereoRig.fromFile('res/2/rigRect.json')

    # Simply rectify two images
    img1_rect, img2_rect = rigRect.rectifyImages(img1, img2)

    # Convert from color(BGR) to gray
    grayR= cv2.cvtColor(img1_rect,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(img2_rect,cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
    dispL= disp
    dispR= stereoR.compute(grayR,grayL)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

    # Using the WLS filter
    filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)
    disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

##    # Resize the image for faster executions
##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

    # Filtering the Results with a closing filter
    closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

    # Colors map
    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN)


# Get 3D points
#points3D = rigRect.get3DPoints(disparityMap)
#ss.points.exportPLY(points3D, "export.ply", img1_rect)

# Normalize and color
#disparityImg = cv2.normalize(src=disparityMap, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)
#cv2.imwrite("disparity.png", disparityImg)

# Show only left image as reference
#cv2.imshow('LEFT rectified', img1_rect)
#cv2.imshow('RIGHT rectified', img2_rect)
#cv2.imshow("Disparity Color", disparityImg)
    cv2.imshow('Filtered Color Depth',filt_Color)

    # Mouse click
    cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)

    print("Press ESC to close.")
#while True:
    if cv2.waitKey(0) == 27:
        break

# Save excel
wb.save("data.xlsx")

cv2.destroyAllWindows()

        
        
