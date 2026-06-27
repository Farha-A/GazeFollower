#!/usr/bin/env python
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import sys
import threading
import time
import traceback

import cv2

from .Camera import Camera  # Adjust import according to your package structure
from ..logger import Log


class WebCamCamera(Camera):
    """
    A class to manage webcam operations, inheriting from the base Camera class.
    """

    def __init__(self, webcam_id=0, img_height=480, img_width=640, cam_fps=30,
                 use_clahe=True, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
        """
        Initializes the WebCamCamera object, sets up the camera properties,
        creates the capture thread, and ensures the save directory exists.

        Attributes:
        ----------
        webcam_id : int
            Which webcam camera is connected.
        cap: cv2.VideoCapture
            The instance of cv2.VideoCapture and it can be None.
        use_clahe : bool
            Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
            the luminance channel of every captured frame before passing it to
            GazeFollower.  Enhances local contrast so iris/pupil boundaries
            stand out without amplifying global noise or shifting colour.
            Defaults to True.
        clahe_clip_limit : float
            Contrast amplification ceiling per tile.  Lower = subtler effect.
        clahe_tile_grid_size : tuple(int, int)
            Neighbourhood size for local histogram computation. (8, 8) is the
            standard default.
        """
        super().__init__()
        self._camera_thread_running = None
        self._camera_thread = None
        self.webcam_id = webcam_id
        self.img_height = img_height
        self.img_width = img_width
        self.cam_fps = cam_fps
        # Build the CLAHE object once; reused every frame (thread-safe read).
        self.use_clahe = use_clahe
        self._clahe = (cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                       tileGridSize=clahe_tile_grid_size)
                       if use_clahe else None)
        self._cap = cv2.VideoCapture()
        # Set the camera resolution and frame rate.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.cam_fps)

    def _create_capture_thread(self):
        """
        Creates and starts a daemon thread for continuously capturing frames from the camera.
        """
        self._camera_thread_running = True
        self._camera_thread = threading.Thread(target=self.capture)
        self._camera_thread.daemon = True
        self._camera_thread.start()

    def capture(self):
        """
        Continuously captures frames from the webcam while the camera is in specific running states.
        If a callback is set, it executes the callback function with the current frame.
        """
        while self._camera_thread_running:
            # Capture a frame from the webcam.
            ret, frame = self._cap.read()
            # Capture the current timestamp.
            timestamp = time.time_ns()
            if not ret:
                Log.w("Failed to grab frame")
                continue

            # Check if the frame is in BGR format (default for OpenCV) and convert to RGB if necessary
            # Preprocessing image data
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to 640x480 if necessary
            frame = cv2.resize(frame, (self.img_width, self.img_height))

            # Optional CLAHE: enhance local contrast on luminance only so
            # iris/pupil edges are crisper without shifting colour or amplifying
            # global noise.  Frame is already RGB here, so RGB2LAB is correct.
            if self.use_clahe:
                lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            # Lock and execute callback function if set.
            try:
                with self.callback_and_param_lock:
                    if self.callback_func is not None:

                        self.callback_func(self.camera_running_state, timestamp, frame, *self.callback_args,
                                           **self.callback_kwargs)

            except Exception as e:
                Log.e(str(e))
                Log.e(f"Traceback:\n{traceback.format_exc()}")

    def open(self):
        """
        Opens the webcam if it is not already opened.
        """
        Log.i("WebCam opened")
        if not self._cap.open(self.webcam_id):
            Log.e("Failed to open webcam camera")
            Log.e(f"Traceback:\n{traceback.format_exc()}")
            raise Exception("Failed to open webcam camera")
        self._create_capture_thread()

    def close(self):
        """
        Releases the webcam resources if the camera is currently opened.
        """
        Log.i("WebCam closed")
        if self._camera_thread is not None:
            self._camera_thread_running = False
            self._camera_thread.join()
        if not self._cap.isOpened():
            self._cap.release()

    def set_on_image_callback(self, func, args=(), kwargs=None):
        """
        Sets a callback function to be called with each captured frame.
        The callback function must have the following args,
            timestamp and frame, which are the timestamp when the image was
            captured and the captured image frame (np.ndarray).

        Parameters:
        - func: The callback function to handle the image frame.
        - args: Tuple of arguments to pass to the callback function.
        - kwargs: Dictionary of keyword arguments to pass to the callback function.
        """
        super().set_on_image_callback(func, args, kwargs)

    def release(self):
        self._camera_thread_running = False
        self._camera_thread.join()
        self.close()
