import threading
import cv2
import time
from detection.pipeline import build_pipeline
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


class OpenCVGstreamerStream:
    """Class that integrates OpenCV with GStreamer for camera streaming on Jetson Orin Nano"""
    
    def __init__(self, camera_id=0, width=1920  , height=1080, fps=30, 
                 sensor_id=0, flip_method=0):
        """
        Initialize the OpenCV-GStreamer camera stream
        
        Args:
            camera_id: Device ID for USB camera (e.g., /dev/video0 would be 0)
            width: Stream width in pixels
            height: Stream height in pixels
            fps: Target frames per second
            sensor_id: Sensor ID for CSI camera (typically 0 or 1 on Jetson)
            flip_method: Video flip method (0=none, 1=counterclockwise, 2=180, 3=clockwise)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.sensor_id = sensor_id
        self.flip_method = flip_method
        
        self.frame = None
        self.lock = threading.Lock()
        self.is_running = False
        self.cap = None
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Check if OpenCV has GStreamer support
        if not self._check_gstreamer_support():
            print("WARNING: OpenCV was not built with GStreamer support!")
            print("GStreamer pipelines will not work with cv2.VideoCapture.")
    
    def _check_gstreamer_support(self):
        """Check if OpenCV was built with GStreamer support"""
        return cv2.getBuildInformation().find("GStreamer") != -1
    
    def build_pipeline_string(self):
            # Simplified CSI camera pipeline
        return (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=1"
        )
    
    def start_capture(self):
        """Start capturing frames using OpenCV with GStreamer pipeline"""
        try:
            pipeline_str = self.build_pipeline_string()
            print(f"Using pipeline: {pipeline_str}")
            
            # Open camera using GStreamer pipeline
            self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                print("Failed to open camera with GStreamer pipeline!")
                return False
            
            # Start capture thread
            self.is_running = True
            capture_thread = threading.Thread(target=self._capture_frames)
            capture_thread.daemon = True
            capture_thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting capture: {e}")
            return False
    
    def _capture_frames(self):
        """Continuously capture frames from the camera using OpenCV"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("Failed to capture frame")
                time.sleep(0.1)  # Prevent CPU spikes on failure
    
    def get_frame(self):
        """Get the latest frame as JPEG encoded bytes"""
        with self.lock:
            if self.frame is None:
                # Return a blank frame if no frame is available
                blank = np.zeros((self.height, self.width, 3), np.uint8)
                _, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()
            
            # Create a copy of the frame to avoid conflicts
            output_frame = self.frame.copy()
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buffer.tobytes()
    
    def generate_frames(self):
        """Generate MJPEG stream from captured frames"""
        while self.is_running:
            frame_bytes = self.get_frame()
            
            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control the frame rate
            time.sleep(1/self.fps)
    
    def process_frame(self, frame, processor_func=None):
        """
        Process a frame with custom processing function
        
        Args:
            frame: The input frame to process
            processor_func: A function that takes a frame and returns a processed frame
        
        Returns:
            The processed frame
        """
        if processor_func is None:
            return frame
        
        return processor_func(frame)
    
    def stop_capture(self):
        """Stop capturing frames and release the camera"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        print("Camera capture stopped")
