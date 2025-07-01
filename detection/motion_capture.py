from enum import Enum
import cv2
import numpy as np
from threading import Lock
from collections import deque
import time

from django.utils import timezone

class TrainState(Enum):
    NO_TRAIN = "no_train"
    TRAIN_ENTERING = "train_entering"
    TRAIN_PASSING = "train_passing"
    TRAIN_EXITING = "train_exiting"


def debug_gpumat(gpu_mat, name):
    """Debug helper to check GpuMat properties"""
    print(f"{name}:")
    print(f"  Empty: {gpu_mat.empty()}")
    print(f"  Size: {gpu_mat.size()}")  # height, width
    print(f"  Type: {gpu_mat.type()}")
    print(f"  Channels: {gpu_mat.channels()}")
    print(f"  Depth: {gpu_mat.depth()}")


DEFAULT_ROI_PERCENTAGES = (
    0,
    0.25,
    1.0,
    0.5
)

class TrainCapture(cv2.VideoCapture):
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
    
    def __init__(self, 
                 *args, 
                background_learning_rate=0.005,
                motion_threshold=30,
                min_contour_area=1000,
                track_roi=False,
                stability_frames=10,
                resize_width=320, 
                debug=False,
                **kwargs):
        # Initialize the parent VideoCapture
        super().__init__(*args, **kwargs)

        #Streams
        self.upload_stream = cv2.cuda.Stream()
        self.background_stream = cv2.cuda.Stream()
        self.diff_stream = cv2.cuda.Stream()
        self.processing_stream = cv2.cuda.Stream()
        
        #GPU Memory Allocation
        self.gpu_current_frame = cv2.cuda_GpuMat()
        self.gpu_resized_frame = cv2.cuda_GpuMat()
        self.gpu_current_gray = cv2.cuda_GpuMat()
        self.gpu_background = cv2.cuda_GpuMat()
        self.gpu_diff = cv2.cuda_GpuMat()
        self.gpu_blurred = cv2.cuda_GpuMat()
        self.gpu_thresh = cv2.cuda_GpuMat()
        self.gpu_morph = cv2.cuda_GpuMat()

        #Parameters
        self.learning_rate = background_learning_rate
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.track_roi = track_roi
        self.stability_frames = stability_frames
        self.resize_width = resize_width
        self.debug = debug

        #State Tracking
        self.train_state = TrainState.NO_TRAIN
        self.state_frame_count = 0
        self.background_initialized = False
        self.frame_count = 0

        # Train detection metrics
        self.motion_history = deque(maxlen=30)  # Last 30 frames
        self.train_start_time = None
        self.train_end_time = None

        # Morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Create Gaussian filter for CUDA
        self.gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC1, cv2.CV_8UC1, (21, 21), 0
        )
        self.morph_close_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_CLOSE, cv2.CV_8UC1, kernel
        )
        self.morph_open_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_OPEN, cv2.CV_8UC1, kernel
)

        # For ROI processing
        if self.track_roi:
            self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = DEFAULT_ROI_PERCENTAGES

        # Lock for thread safety
        self.lock = Lock()
    
    @property
    def train_detected(self):
        """Check if a train is currently detected"""
        return self.train_state != TrainState.NO_TRAIN

    @property
    def motion_percentage(self):
        """Calculate the percentage of motion in the current frame"""
        return self.motion_history[-1] if self.motion_history else 0.0

    def process_frame(self, frame):
        """Process a single frame and return train detection results"""
        
        # Stage 1: Upload and convert to grayscale
        self.gpu_current_frame.upload(frame, self.upload_stream)
        self.upload_stream.waitForCompletion()
        
        # Resize frame for faster processing
        original_height, original_width = frame.shape[:2]
        resize_height = int(original_height * (self.resize_width / original_width))

        self.gpu_resized_frame = cv2.cuda.resize(self.gpu_current_frame, (self.resize_width, resize_height), 
                        stream=self.background_stream)

        self.gpu_current_gray = cv2.cuda.cvtColor(self.gpu_current_frame, cv2.COLOR_BGR2GRAY, stream=self.background_stream)
        
        # Stage 2: Background model comparison (BEFORE updating)
        if not self.background_initialized:
            self.gpu_background = self.gpu_current_gray.copyTo(self.background_stream)
            self.background_initialized = True
            return self._create_result(frame, 0, [])
        
        # Compute difference against CURRENT background (before updating)
        self.background_stream.waitForCompletion()
        self.gpu_diff = cv2.cuda.absdiff(self.gpu_current_gray, self.gpu_background, stream=self.diff_stream)

        DEAD_ZONE = 3  # Ignore differences < 3 pixel values
        _, self.gpu_diff = cv2.cuda.threshold(self.gpu_diff, DEAD_ZONE, 0, cv2.THRESH_TOZERO, stream=self.diff_stream)

        # Stage 3: Update background model AFTER comparison
        self.gpu_background = cv2.cuda.addWeighted(
            self.gpu_background, 1.0 - self.learning_rate,
            self.gpu_current_gray, self.learning_rate,
            0, stream=self.background_stream
        )

        # # Stage 4: Noise reduction and thresholding
        # # Use the Gaussian filter we created
        self.gpu_blurred = self.gaussian_filter.apply(self.gpu_diff, 
                                  stream=self.processing_stream)
        
        _, self.gpu_thresh = cv2.cuda.threshold(self.gpu_blurred, 
                          self.motion_threshold, 255, cv2.THRESH_BINARY,
                          stream=self.processing_stream)

        # Morphological operations to clean up detection
        self.gpu_morph = self.morph_close_filter.apply(self.gpu_thresh, stream=self.processing_stream)
        self.gpu_thresh = self.morph_open_filter.apply(self.gpu_morph, stream=self.processing_stream)

        # Download processed frame for contour analysis
        self.processing_stream.waitForCompletion()
        motion_mask = self.gpu_thresh.download()
        
        # Find contours and analyze
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        significant_contours = [c for c in contours 
                              if cv2.contourArea(c) > self.min_contour_area]
        
        if self.debug and self.frame_count % 300 == 0:
            self.save_debug_images(self.frame_count)

        # Calculate motion metrics
        total_motion_area = sum(cv2.contourArea(c) for c in significant_contours)
        motion_percentage = (total_motion_area / (motion_mask.shape[0] * motion_mask.shape[1]))
        
        # Update motion history
        self.motion_history.append(motion_percentage)
        
        # Update train state
        self._update_train_state(motion_percentage, significant_contours)
        
        self.frame_count += 1
        
        return self._create_result(frame, motion_percentage, significant_contours)
    
    def _update_train_state(self, motion_percentage, contours):
        """Update train detection state based on motion analysis"""
        
        # Define motion thresholds
        HIGH_MOTION_THRESHOLD = 0.05  # Percentage of frame with motion
        LOW_MOTION_THRESHOLD = 0.01
        
        # Determine current frame state
        if motion_percentage > HIGH_MOTION_THRESHOLD and len(contours) > 0:
            frame_state = "high_motion"
        elif motion_percentage > LOW_MOTION_THRESHOLD:
            frame_state = "low_motion"
        else:
            frame_state = "no_motion"
        
        # State machine logic
        if self.train_state == TrainState.NO_TRAIN:
            if frame_state == "high_motion":
                self.state_frame_count += 1
                if self.state_frame_count >= self.stability_frames:
                    self.train_state = TrainState.TRAIN_ENTERING
                    self.train_start_time = time.time()
                    self.state_frame_count = 0
            else:
                self.state_frame_count = 0
        
        elif self.train_state == TrainState.TRAIN_ENTERING:
            if frame_state in ["high_motion", "low_motion"]:
                self.state_frame_count += 1
                if self.state_frame_count >= self.stability_frames:
                    self.train_state = TrainState.TRAIN_PASSING
                    self.state_frame_count = 0
            else:
                # False alarm - no sustained motion
                self.train_state = TrainState.NO_TRAIN
                self.state_frame_count = 0
        
        elif self.train_state == TrainState.TRAIN_PASSING:
            if frame_state == "no_motion":
                self.state_frame_count += 1
                if self.state_frame_count >= self.stability_frames:
                    self.train_state = TrainState.TRAIN_EXITING
                    self.state_frame_count = 0
            else:
                self.state_frame_count = 0  # Reset if motion continues
        
        elif self.train_state == TrainState.TRAIN_EXITING:
            if frame_state == "no_motion":
                self.state_frame_count += 1
                if self.state_frame_count >= self.stability_frames * 2:  # More stability for exit
                    self.train_state = TrainState.NO_TRAIN
                    self.train_end_time = time.time()
                    self.state_frame_count = 0
            else:
                # Train still present
                self.train_state = TrainState.TRAIN_PASSING
                self.state_frame_count = 0

    def _create_result(self, frame, motion_percentage, contours):
        """Create result dictionary with detection information"""
        
        result = {
            'frame': frame,
            'train_detected': self.train_state != TrainState.NO_TRAIN,
            'train_state': self.train_state.value,
            'motion_percentage': motion_percentage,
            'contour_count': len(contours),
            'frame_count': self.frame_count,
            'train_duration': None
        }
        
        # Calculate train duration if available
        if self.train_start_time and self.train_end_time:
            result['train_duration'] = self.train_end_time - self.train_start_time
        elif self.train_start_time and self.train_state != TrainState.NO_TRAIN:
            result['train_duration'] = time.time() - self.train_start_time
        
        return result

    def save_debug_images(self, frame_num):
        """Save debug images for analysis"""
        debug_dir = 'img/train_debug'
        import os
        os.makedirs(debug_dir, exist_ok=True)

        # Current frame (ROI)
        current_cpu = self.gpu_current_frame.download()
        cv2.imwrite(f'{debug_dir}/current_{frame_num:04d}.png', current_cpu)
        
        # Background
        bg_cpu = self.gpu_background.download()
        cv2.imwrite(f'{debug_dir}/background_{frame_num:04d}.png', bg_cpu)
        
        # Raw difference
        diff_cpu = self.gpu_diff.download()
        cv2.imwrite(f'{debug_dir}/difference_{frame_num:04d}.png', diff_cpu)
        
        # Thresholded result
        thresh_cpu = self.gpu_thresh.download()
        cv2.imwrite(f'{debug_dir}/threshold_{frame_num:04d}.png', thresh_cpu)
        
        # print(f"Debug images saved to {debug_dir} for frame {frame_num}")


    def get_detection_summary(self):
        """Get summary of recent detection activity"""
        recent_motion = list(self.motion_history)[-10:] if self.motion_history else []
        avg_motion = np.mean(recent_motion) if recent_motion else 0
        
        return {
            'current_state': self.train_state.value,
            'average_recent_motion': avg_motion,
            'frames_processed': self.frame_count,
            'background_initialized': self.background_initialized
        }
    
    def read(self, ignore_motion=False):
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
            self.frame_count += 1
            if ret and not ignore_motion:
                if self.track_roi:
                    roi = frame.copy()[int(frame.shape[0] * self.roi_y1):int(frame.shape[0] * self.roi_y2),
                                      int(frame.shape[1] * self.roi_x1):int(frame.shape[1] * self.roi_x2)]
                else:
                    roi = frame.copy()  
                
                self.process_frame(roi)
            
            return ret, frame
    
