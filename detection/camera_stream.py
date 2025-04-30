import threading
import cv2
import time
import numpy as np


class CameraStream:
    """Class to handle camera streaming functionality"""
    
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.frame = None
        self.lock = threading.Lock()
        self.is_running = False
    
    def setup_camera(self):
        """Initialize the camera with the specified settings"""
        self.camera = cv2.VideoCapture(self.camera_id)
        
        # Try to set camera resolution and FPS
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Check if camera opened successfully
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            return False
        
        # Get actual camera settings (may differ from requested)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized with resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
        return True
    
    def start_capture(self):
        """Start capturing frames from the camera in a separate thread"""
        if self.setup_camera():
            self.is_running = True
            capture_thread = threading.Thread(target=self._capture_frames)
            capture_thread.daemon = True
            capture_thread.start()
            return True
        return False
    
    def _capture_frames(self):
        """Continuously capture frames from the camera"""
        while self.is_running:
            success, img = self.camera.read()
            if success:
                with self.lock:
                    self.frame = img
            else:
                print("Failed to capture image")
                time.sleep(0.1)  # Prevent CPU spikes on failure
    
    def get_frame(self):
        """Get the latest frame as JPEG encoded bytes"""
        with self.lock:
            if self.frame is None:
                # Return a blank frame if no frame is available
                blank = np.zeros((self.height, self.width, 3), np.uint8)
                _, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()
            
            # Create a copy of the frame to avoid conflicts with the capture thread
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
    
    def stop_capture(self):
        """Stop capturing frames and release the camera"""
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
