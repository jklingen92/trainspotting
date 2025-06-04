import cv2
import numpy as np
from threading import Lock

from django.utils import timezone

class MotionCapture(cv2.VideoCapture):
    """
    A subclass of OpenCV's VideoCapture that performs motion detection on frames as they're read.
    
    Parameters:
    -----------
    source : int or str
        Camera index or video file path, same as cv2.VideoCapture
    skip_frames : int, optional (default=5)
        Number of frames to skip between motion detection operations
    threshold : int, optional (default=25)
        Threshold for pixel differences (0-255)
    min_area_start : float, optional (default=0.05)
        Minimum percentage of frame that must change for motion detection
    resize_width : int, optional (default=480)
    min_area_end : float, optional (default=0.05)
        Minimum percentage of frame that must change for motion detection
    resize_width : int, optional (default=480)
        Width to resize frames to for processing
    fps : int, optional (default=30)
        Frames per second (used for time calculations)
    min_motion_seconds : float, optional (default=1.0)
        Minimum motion duration to be considered valid
    min_stillness_seconds : float, optional (default=1.0)
        Minimum stillness duration to be considered valid
    use_cuda : bool, optional (default=False)
        Whether to use CUDA acceleration if available
    """
    
    def __init__(self, *args, skip_frames=5, threshold=10, min_area_start=0.2, min_area_end=0.05,
                 resize_width=360, fps=30, min_motion_seconds=1.0, 
                 min_stillness_seconds=1.0, use_cuda=False, 
                 learning_frames=150, **kwargs):
        # Initialize the parent VideoCapture
        super().__init__(*args, **kwargs)
        
        # Store parameters
        self.skip_frames = skip_frames
        self.threshold = threshold
        self.min_area_start = min_area_start
        self.min_area_end = min_area_end
        self.resize_width = resize_width
        self.fps = fps
        self.min_motion_frames = int(min_motion_seconds * fps)
        self.min_stillness_frames = int(min_stillness_seconds * fps)
        self.use_cuda = False
        self.learning_frames = learning_frames
        
        if use_cuda:
            if not cv2.cuda.getCudaEnabledDeviceCount() > 0:
                raise ValueError("CUDA is not available on this system.")
            else:
                self.use_cuda = True
        
        # Initialize motion detection variables
        self.frame_count = 0
        self.background = None
        self.motion_detected = False
        self.consecutive_motion_frames = 0
        self.consecutive_still_frames = 0
        self.motion_start_time = None
        self.motion_percentage = 0.0
        self.last_motion_time = None
        
        # Lock for thread safety
        self.lock = Lock()
    
    def _preprocess_frame(self, frame):
        """Preprocess the frame for motion detection"""
        if frame is None:
            return None
            
        if self.use_cuda:
            # GPU-based preprocessing
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            resized_gpu = cv2.cuda.resize(gpu_frame, (self.resize_width, int(frame.shape[0] * (self.resize_width / frame.shape[1]))))
            gray_gpu = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_gpu.download(), (5, 5), 0)
        
        else:
            # CPU-based preprocessing
            resized_frame = cv2.resize(frame, (self.resize_width, int(frame.shape[0] * (self.resize_width / frame.shape[1]))))
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        return blurred_frame
    
    def _detect_motion(self, current_frame):
        """
        Detect motion by comparing the current frame with the background
        
        Returns:
        --------
        motion_percentage : float
            Percentage of pixels that changed between frames
        """
        # Convert to grayscale and resize for faster processing
        current = self._preprocess_frame(current_frame)
        
        # Initialize background on first frame
        if self.background is None:
            self.background = current
            return 0.0
        
        # Compute absolute difference between current frame and background
        if self.use_cuda:
            # GPU-based absolute difference
            # Placeholder - actual implementation would use cv2.cuda.absdiff
            # Upload to GPU
            gpu_current = cv2.cuda_GpuMat()
            gpu_background = cv2.cuda_GpuMat()
            gpu_current.upload(current)
            gpu_background.upload(self.background)
            
            # Process on GPU
            gpu_delta = cv2.cuda.absdiff(gpu_current, gpu_background)
            gpu_thresh = cv2.cuda.threshold(gpu_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Download result (only for counting)
            thresh = gpu_thresh.download()
        else:
            # CPU-based absolute difference
            frame_delta = cv2.absdiff(current, self.background)
            thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Calculate percentage of changed pixels
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_percentage = changed_pixels / total_pixels
        
        # Update background (simple approach - could use more sophisticated background subtraction)
        alpha = 0.05  # Learning rate
        
        if self.use_cuda:
            # GPU-based background update
            # Placeholder - actual implementation would use cv2.cuda.addWeighted
            cv2.cuda.addWeighted(gpu_background, 1 - alpha, gpu_current, alpha, 0, gpu_background)
            self.background = gpu_background.download()
        else:
            # CPU-based background update
            self.background = cv2.addWeighted(self.background, 1 - alpha, current, alpha, 0)
        
        return motion_percentage
    
    def _update_motion_state(self, motion_percentage):
        """Update motion detection state based on current frame analysis"""
        # Check if motion percentage exceeds the minimum area threshold
        if not self.motion_detected:
            is_motion = motion_percentage >= self.min_area_start
        else:
            is_motion = motion_percentage >= self.min_area_end
        
        # Update consecutive frame counters
        if is_motion:
            self.consecutive_motion_frames += self.skip_frames
            self.consecutive_still_frames = 0
            
            # Record motion start time when we first detect motion
            if not self.motion_detected and self.consecutive_motion_frames >= self.min_motion_frames:
                self.motion_detected = True
                self.motion_start_time = self.last_motion_time = timezone.localtime()
        else:
            self.consecutive_still_frames += self.skip_frames
            self.consecutive_motion_frames = 0
            
            # Reset motion flag if stillness persists long enough
            if self.motion_detected and self.consecutive_still_frames >= self.min_stillness_frames:
                self.motion_detected = False
        
        # Store current motion percentage
        self.motion_percentage = motion_percentage
        
        # Update last motion time if motion is detected
        if is_motion:
            self.last_motion_time = timezone.localtime()
    
    def read(self):
        """
        Read the next frame from the video source and perform motion detection
        
        Returns:
        --------
        ret : bool
            Whether the frame was successfully read
        frame : numpy.ndarray
            The frame read from the video source
        """
        with self.lock:
            # Call the parent read method to get the frame
            ret, frame = super().read()
            
            if ret:
                self.frame_count += 1
                
                if self.learning_frames > 0:
                    self.learning_frames -= 1
                # Perform motion detection only on specified frames to reduce CPU usage
                elif self.frame_count % self.skip_frames == 0:
                    motion_percentage = self._detect_motion(frame.copy())
                    self._update_motion_state(motion_percentage)
            
            return ret, frame
    
    @property
    def motion_duration(self):
        """Return the duration of the current motion in seconds"""
        if self.motion_detected and self.motion_start_time is not None:
            return (timezone.localtime() - self.motion_start_time).total_seconds()
        return 0.0
    
    @property
    def time_since_last_motion(self):
        """Return time in seconds since last motion was detected"""
        if self.last_motion_time is not None:
            return (timezone.localtime() - self.last_motion_time).total_seconds()
        return float('inf')  # No motion detected yet
