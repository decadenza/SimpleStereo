import sys
import os

import cv2

import simplestereo as ss
"""
Simultaneously capture left-right image pairs and save into folder.
"""

# These are the ids of the video capturing device to open indexes.
# They depends on YOUR hardware (usually the internal webcam is 0).
LEFT_CAMERA = 2
RIGHT_CAMERA = 0

# Export path
curPath = os.path.dirname(os.path.realpath(__file__))
savePath = os.path.join(curPath,"res","new")

print("Saving in:", savePath)
print("Press S to save current pair, ESC to exit.")


with ss.utils.Capture(LEFT_CAMERA) as cap1, ss.utils.Capture(RIGHT_CAMERA) as cap2:
    
    # Set HD resolution (if supported by your cameras)
    cap1.setResolution(1280,720) # Returns True if settings is ok
    cap2.setResolution(1280,720) # False otherwise
    
    # Other properties may be set using OpenCV object directly
    #print(cap1.video_capture.set(cv2.CAP_PROP_CONTRAST, 80))
    #print(cap2.video_capture.set(cv2.CAP_PROP_CONTRAST, 80))
    
    cap1.start()
    cap2.start()
    
    i=0
    while(True):
        img1 = cap1.get()
        img2 = cap2.get()
        if img1 is None or img2 is None: # Camera not ready...
            continue
        
        cv2.imshow("Left", img1)
        cv2.imshow("Right", img2)
        c = cv2.waitKey(100)             # Wait 100 ms
        
        if c == 27:
            break
        elif c == 115 or c == 83:
            print(i, "saved!")
            cv2.imwrite(os.path.join(savePath, str(i)+"_L.png"), img1)
            cv2.imwrite(os.path.join(savePath, str(i)+"_R.png"), img2)
            i+=1

cv2.destroyAllWindows()
print("Capture finished!")
