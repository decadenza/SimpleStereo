#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
from time import sleep

import numpy as np
import cv2

import simplestereo as ss
'''
Project pattern and capture object photos.

This is a *DEMO* and requires some tweaks to work on your system!
'''     
   
CAMERA = 0 # Depends on *YOUR* hardware (usually the internal webcam is 0).
MAIN_MONITOR_RESOLUTION = (1980,1080)
PROJECTOR_RESOLUTION = (1280,720)

# Export paths
CURPATH = os.path.dirname(os.path.realpath(__file__))
SAVEPATH = os.path.join(CURPATH,"scannedObjects")

# Fringe image path
FRINGE_IMG = os.path.join(CURPATH, "res", "stereoFTP","fringe8.png")


if __name__ == "__main__":

    # Initialization
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)

    print("New images will be saved in:", SAVEPATH)
    print("Press S to save current image, ESC to exit.")

    with ss.utils.Capture(CAMERA) as cap:
        
        print("CAMERA INITIALIZATION...")
        # CODEC
        codec = int(cap.video_capture.get(cv2.CAP_PROP_FOURCC))
        print("CAP_PROP_FOURCC get:", codec.to_bytes((codec.bit_length() + 7) // 8, 'big').decode()[::-1] )
        print("CAP_PROP_FOURCC set", cap.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MPEG')))
        codec = int(cap.video_capture.get(cv2.CAP_PROP_FOURCC))
        print("CAP_PROP_FOURCC get:", codec.to_bytes((codec.bit_length() + 7) // 8, 'big').decode()[::-1] )
        
        # FPS
        print("CAP_PROP_FPS get",cap.video_capture.get(cv2.CAP_PROP_FPS))
        print("CAP_PROP_FPS set",cap.video_capture.set(cv2.CAP_PROP_FPS, 30))
        print("CAP_PROP_FPS get",cap.video_capture.get(cv2.CAP_PROP_FPS))
        
        # Set resolution (if supported by your cameras)
        print("Get resolution", cap.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), cap.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Set resolution", cap.setResolution(1920,1080)) # Returns True if settings is ok
        print("Get resolution", cap.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), cap.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.start()
        winW = int(MAIN_MONITOR_RESOLUTION[0]*0.41)
        winH = int(MAIN_MONITOR_RESOLUTION[1]*0.41)
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Camera", 0, 0);
        cv2.resizeWindow("Camera", winW, winH)
        
        # Show saved image
        #cv2.namedWindow("SAVED", cv2.WINDOW_NORMAL)
        #cv2.moveWindow("SAVED", 0, winH+49);
        #cv2.resizeWindow("SAVED", winW, winH)
        
        # Fullscreen projector on the right
        cv2.namedWindow("Projector", cv2.WINDOW_FREERATIO)
        cv2.moveWindow("Projector", MAIN_MONITOR_RESOLUTION[0], 0);
        cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while(True):
            CURRENT_OBJ = input("Current object name: ")
            
            ### PREVIEW
            print("Press ENTER to start capture or ESC to cancel")
            
            # Project white
            projImg = cv2.imread(FRINGE_IMG)
            cv2.imshow("Projector", projImg)
            
            # Give time to setup camera
            while(True):
                img = cap.get()
                
                if img is None: # Camera not ready...
                    continue
                
                cv2.imshow("Camera", img)
                c = cv2.waitKey(100)             # Wait ms
                
                if c == 13: # ENTER
                    cv2.imwrite(os.path.join(SAVEPATH, CURRENT_OBJ+".png"), img)
                    print(CURRENT_OBJ, "saved!")
                    #cv2.imshow("SAVED", img)
                    cv2.waitKey(100)
                    break
                    
                elif c == 27: # ESC
                    break
            

    cv2.destroyAllWindows()

