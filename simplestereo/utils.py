"""
utils
==========
This module provides general utilities.
"""
import os
from threading import Thread

import numpy as np
import cv2


class Capture:
    """Capture a video stream continuously.
    
    Allows to capture a video stream in a separate thread and grab the current frame when needed, 
    minimizing lags for streaming and webcams.
    It supports webcams, URL streaming and video files.
    
    Parameters
    ----------
    device : int or str
        Id of the opened video capturing device (i.e. a camera index). If there is a single camera connected, usually it will be 0 (default 0).
        Also the string containing full URL of the video stream (e.g. *protocol://username:password@script?params*) or a path to a video file.
    flipY : bool
        If True, output image is flipped on the Y-axis. Default to False.
        
    Raises
    ------
    ValueError
        If device cannot be opened.
    
    .. note::
        When using video streaming URL the initialization may require some time. Check that the image frame is not None
        and / or insert a time.sleep() after object constructor.
        
    Todo
    ----
    Add support for other options (e.g. change focal length where supported).
    """
    def __init__(self, device = 0, flipY=False):
        # Check if we're opening a video file
        self.isFile = os.path.isfile(device)
        self.video_capture = cv2.VideoCapture(device)
        # Flip around y axis if needed
        self.flip = cv2.flip if flipY else lambda f, *a, **k: f 
        self.running = False
        # Keep this as attribute (needed for streaming)
        self.frame = None   
        
        if self.isFile:                             # If we are opening a video file
            self.getFrame = self.video_capture.read # we use read()
            self.grab = lambda *args: None          # and cannot grab()
        else:                                       # If opening a webcam or a URL streaming
            self.grab = self.video_capture.grab     # that grabs continuously (grab() is fast)
            self.getFrame = self.video_capture.retrieve # and retrieve() (slower) the image only when needed
            self.t=Thread(target=self.__loop)       # use a separate thread
            self.t.daemon=True                          
        
        # Check if capture is opened
        if self.video_capture is None or not self.video_capture.isOpened():
            raise ValueError('Cannot open device!')
            
    
    def __del__(self):
        self.stop()
        self.video_capture.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        # To allow use in with statement
        self.__del__()
          
    def start(self):
        """
        Start the capture in a separate thread.
        
        No need to call this for video files.
        The thread continuously calls ``grab()`` to stay updated to the last frame,
        while ``retrieve()`` is called only when the frame is actually needed.
        """
        if self.isFile: # Do not start if it is file
            return
        self.running = True
        self.t.start()
        
    
    def stop(self):
        """
        Stop the capture.
        
        When finished, remember to call this method to **stop the capturing thread**.
        No need to call this for video files.
        
        """
        if self.isFile: # N/A if is a file
            return
        if self.running:
            self.running = False
            self.t.join()
        return
        
    def __loop(self):
        while(self.running):
            self.grab()
            
    def get(self):
        """
        Retrieve the current frame.
        
        Returns None if there is no frame (e.g. end of video file).
        """
        ret, self.frame = self.getFrame()
        if not ret:
            return None
        return self.flip(self.frame, 1)
    
    def setResolution(self, width, height):
        """
        Set resolution of the camera.
        
        Do not call this for video files or when the thread is running.
        It works only for supported cameras.
        
        Parameters
        ----------
        width, heigth : int
            Width and height to be set in pixels.
        
        Returns
        -------
        bool
            True if the resolution was set successfully, False otherwise.
        """
        # You cannot change resolution while running or on files
        if self.running or self.isFile:
            return False
        
        # Set properties. Each returns === True on success.
        return self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width) and self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def setFrameRate(self, fps):
        """
        Set framerate of the camera.
        
        Do not call this for video files or when the thread is running.
        It works only for supported cameras.
        
        Parameters
        ----------
        fps : int
            Frames per second.
        
        Returns
        -------
        bool
            True if the framerate was set successfully, False otherwise.
        """
        # You cannot change resolution while running or on files
        if self.running or self.isFile:
            return False
        
        return self.video_capture.set(cv2.CAP_PROP_FPS, fps)



def moveExtrinsicOriginToFirstCamera(R1,R2,t1,t2):
    """
    Center extrinsic parameters world coordinate system into camera 1.
    
    Compute R (rotation from camera 1 to camera 2) and T (translation from camera 1 to camera 2) as used in OpenCV
    from extrinsic of two cameras centered anywhere else.
    This is particularly useful when the world coordinate system is not centered into the first camera.
    
    Parameters
    ----------
    R1, R2 : np.array
        3x3 rotation matrices that go from world origin to each camera center.
    t1, t2 : np.array
        3x1 translation vectors that go from world origin to each camera center.
        
    Returns
    -------
    numpy.ndarray
        Rotation matrix between the 1st and the 2nd camera coordinate systems as numpy.ndarray.
    numpy.ndarray
        Translation vector between the coordinate systems of the cameras as numpy.ndarray.
    """
    t1 = t1.reshape((-1,1)) # Force vertical shape
    t2 = t2.reshape((-1,1))
    R = R2.dot(R1.T)
    t = t2 - R2.dot(R1.T).dot(t1)
    return R, t
    




def getCrossProductMatrix(v):
    """
    Build the 3x3 antisymmetric matrix representing the cross product with v.
    
    In literature this is often indicated as [v]\ :subscript:`x`.
    
    Parameters
    ----------
    v : numpy.ndarray or list
        A 3-dimensional vector.
    
    Returns
    -------
    numpy.ndarray
        A 3x3 matrix representing the cross product with the input vector.
    """
    v = v.ravel()
    return np.array( [ [0, -v[2], v[1]], \
                       [v[2], 0, -v[0]], \
                       [-v[1], v[0], 0] ] , dtype=np.float)







def drawCorrespondingEpipolarLines(img1, img2, F, x1=[], x2=[], color=(0,0,255), thickness=1):
    """
    Draw epipolar lines passing by given coordinates in img1 or img1.
    
    The epipolar lines can be drawn on the images, knowing the
    fundamental matrix.
    Please note that this is an in-place method, i.e. passed images will
    be modified directly.
    Distortion is *not* taken into account.
    
    Parameters
    ----------
    img1, img2 : cv2.Mat
        A couple of OpenCV images taken with a stereo rig (ordered).
    F : numpy.ndarray
        3x3 fundamental matrix.
    x1, x2 : list
        List of (x,y) coordinate points on the image 1 (or image 2, respectively).
    color : tuple, optional
        Color as BGR tuple (default to (0,0,255) (red)).
    thickness : int, optional
        Thickness of the lines in pixels (default to 1).
    
    Returns
    -------
    None
    
    
    .. note::
        This function needs *undistorted* images.
    """
    def drawLineOnImg1(line):
        nonlocal img1
        if line[1] == 0: # Vertical line
            line_from = ( int(round( (-line[2]/line[0])[0])), 0 )
            line_to = ( int(round( (-line[2]/line[0])[0])), img1.shape[0] )
        else:
            line_from = ( 0, int(round( (-line[2]/line[1])[0])) )
            line_to = ( img2.shape[1], int(round( (-(line[0]*img1.shape[1] + line[2])/line[1])[0])) )
        cv2.line(img1, (line_from[0],line_from[1]), (line_to[0],line_to[1]) , color=color, thickness=thickness )
        return ((line_from[0]+line_to[0])/2,(line_from[1]+line_to[1])/2)
    
    def drawLineOnImg2(line):
        nonlocal img2
        if line[1] == 0: # Vertical line
            line_from = ( int(round( (-line[2]/line[0])[0])), 0 )
            line_to = ( int(round( (-line[2]/line[0])[0])), img2.shape[0] )
        else:
            line_from = ( 0, int(round( (-line[2]/line[1])[0])) )
            line_to = ( img2.shape[1], int(round( (-(line[0]*img2.shape[1] + line[2])/line[1])[0])) )
        cv2.line(img2, (line_from[0],line_from[1]), (line_to[0],line_to[1]) , color=color, thickness=thickness )
        return ((line_from[0]+line_to[0])/2,(line_from[1]+line_to[1])/2)
        
    # Compute lines corresponding to x1 points
    for x in x1:
        p = np.array([ [x[0]], [x[1]], [1]])
        line = F.dot(p) # Find epipolar line on img2 (homogeneous coordinates)
        k = drawLineOnImg2(line)
        line = F.T.dot(np.array([ [k[0]], [k[1]], [1]])) # Find epipolar line on img1 (homogeneous coordinates)
        drawLineOnImg1(line)
        
    # Compute lines corresponding to x2 points
    for x in x2:
        p = np.array([ [x[0]], [x[1]], [1]])
        line = F.T.dot(p) # Find epipolar line on img1 (homogeneous coordinates)
        k = drawLineOnImg1(line)
        line = F.dot(np.array([ [k[0]], [k[1]], [1]])) # Find epipolar line on img2 (homogeneous coordinates)
        p = drawLineOnImg2(line)
        



