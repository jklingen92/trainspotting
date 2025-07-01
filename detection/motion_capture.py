from enum import Enum
import cv2
import numpy as np
from threading import Lock
from collections import deque
import time

from django.utils import timezone

<<<<<<< Updated upstream
def preprocess_frame(frame, resize_width=480, use_cuda=False):
    """Preprocess the frame for motion detection"""
    if frame is None:
        return None
        
    if use_cuda:
        # GPU-based preprocessing
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        resized_gpu = cv2.cuda.resize(gpu_frame, (resize_width, int(frame.shape[0] * (resize_width / frame.shape[1]))))
        gray_gpu = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_gpu.download(), (5, 5), 0)
    
    else:
        # CPU-based preprocessing
        resized_frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * (resize_width / frame.shape[1]))))
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    return blurred_frame


def get_motion_percentage(frame_1, frame_2, threshold=10, use_cuda=False):
    """
    Detect motion by comparing the current frame with the background
    
    Returns:
    --------
    motion_percentage : float
        Percentage of pixels that changed between frames
    """

    gpu_1 = gpu_2 = None

    # Compute absolute difference between current frame and background
    if use_cuda:
        # GPU-based absolute difference
        # Placeholder - actual implementation would use cv2.cuda.absdiff
        # Upload to GPU
        gpu_1 = cv2.cuda_GpuMat()
        gpu_2 = cv2.cuda_GpuMat()
        gpu_1.upload(frame_1)
        gpu_2.upload(frame_2)
        
        # Process on GPU
        gpu_delta = cv2.cuda.absdiff(gpu_1, gpu_2)
        gpu_thresh = cv2.cuda.threshold(gpu_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Download result (only for counting)
        thresh = gpu_thresh.download()
    else:
        # CPU-based absolute difference
        frame_delta = cv2.absdiff(frame_1, frame_2)
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Calculate percentage of changed pixels
    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    motion_percentage = changed_pixels / total_pixels

    return motion_percentage, gpu_1, gpu_2

class MotionCapture(cv2.VideoCapture):
=======
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
>>>>>>> Stashed changes
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
                
<<<<<<< Updated upstream
                if self.learning_frames > 0:
                    self.learning_frames -= 1
                # Perform motion detection only on specified frames to reduce CPU usage
                elif self.frame_count % self.skip_frames == 0:
                    # Preprocess the frame 
                    current_frame = preprocess_frame(frame, resize_width=self.resize_width, use_cuda=self.use_cuda)

                    motion_percentage, gpu_frame, gpu_background = get_motion_percentage(
                        current_frame,
                        self.background,
                        threshold=self.threshold,
                        use_cuda=True
                    )

                    # Update background (simple approach - could use more sophisticated background subtraction)
                    alpha = 0.05  # Learning rate
                    
                    if self.use_cuda:
                        # GPU-based background update
                        cv2.cuda.addWeighted(gpu_background, 1 - alpha, gpu_frame, alpha, 0)
                        self.background = gpu_background.download()
                    else:
                        # CPU-based background update
                        self.background = cv2.addWeighted(self.background, 1 - alpha, current_frame, alpha, 0)
    
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


def check_for_train(video_file: str, n_frames: int = None, threshold=10, use_cuda=True):
    """
    Detects train passage by analyzing motion patterns in three frames (beginning, middle, end) of a video file.
    
    Args:
        video_file: Path to the video file
        n_frames: Total number of frames in the video. If None, will be determined automatically (default: None)
        threshold: Base threshold for motion detection (default: 10)
        use_cuda: Whether to use CUDA acceleration (default: True)
    
    Returns:
        True: Clear train passage detected (empty-train-empty pattern)
        False: No train detected (empty-empty-empty pattern)
        None: Unclear/anomalous pattern detected or video processing error
    """
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_file}")
            return None
        
        # Determine frame count if not provided
        if n_frames is None:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate frame count
        if n_frames < 3:
            print(f"Error: Video has insufficient frames ({n_frames})")
            cap.release()
            return None
        
        # Calculate frame positions (beginning, middle, end)
        start_frame_idx = 0
        middle_frame_idx = n_frames // 2
        end_frame_idx = n_frames - 1
        
        frames = {}
        
        # Extract the three key frames
        for frame_name, frame_idx in [('start', start_frame_idx), ('middle', middle_frame_idx), ('end', end_frame_idx)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read {frame_name} frame at index {frame_idx}")
                cap.release()
                return None
            frames[frame_name] = frame
        
        cap.release()
        
        # Preprocess all frames
        start_processed = preprocess_frame(frames['start'], use_cuda=use_cuda)
        middle_processed = preprocess_frame(frames['middle'], use_cuda=use_cuda)
        end_processed = preprocess_frame(frames['end'], use_cuda=use_cuda)
        
        # Calculate motion between consecutive frames
        start_to_middle_motion, _, _ = get_motion_percentage(
            start_processed, middle_processed, threshold=threshold, use_cuda=use_cuda
        )
        
        middle_to_end_motion, _, _ = get_motion_percentage(
            middle_processed, end_processed, threshold=threshold, use_cuda=use_cuda
        )
        
        # Also check start to end for baseline stability
        start_to_end_motion, _, _ = get_motion_percentage(
            start_processed, end_processed, threshold=threshold, use_cuda=use_cuda
        )
        
        # Define adaptive thresholds based on the base threshold
        baseline_threshold = max(threshold * 0.5, 3)  # 50% of threshold, min 3
        train_threshold = threshold * 1.5  # 150% of threshold for clear train detection
        
        # Pattern analysis
        # Case 1: Train passage (empty-train-empty)
        # Expected: High motion start->middle and middle->end, low motion start->end
        if (start_to_middle_motion >= train_threshold and 
            middle_to_end_motion >= train_threshold and 
            start_to_end_motion <= baseline_threshold):
            return True
        
        # Case 2: No train (empty-empty-empty)  
        # Expected: Low motion across all comparisons
        if (start_to_middle_motion <= baseline_threshold and
            middle_to_end_motion <= baseline_threshold and
            start_to_end_motion <= baseline_threshold):
            return False
        
        # Case 3: Anomalous patterns
        return None
        
    except Exception as e:
        print(f"Error processing video {video_file}: {str(e)}")
        return None


def check_for_train_with_diagnostics(video_file: str, n_frames: int = None, threshold=10, use_cuda=True):
    """
    Enhanced version that returns diagnostic information along with the detection result.
    
    Args:
        video_file: Path to the video file
        n_frames: Total number of frames in the video. If None, will be determined automatically (default: None)
        threshold: Base threshold for motion detection (default: 10)
        use_cuda: Whether to use CUDA acceleration (default: True)
    
    Returns:
        tuple: (detection_result, diagnostics_dict)
        
    diagnostics_dict contains:
        - motion_values: dict with all motion percentages
        - pattern_analysis: description of detected pattern
        - thresholds_used: dict of threshold values
        - video_info: basic video information
    """
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return None, {"error": f"Could not open video file {video_file}"}
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use provided n_frames or auto-detected
        if n_frames is None:
            n_frames = total_frames
        
        if n_frames < 3:
            cap.release()
            return None, {"error": f"Video has insufficient frames ({n_frames})"}
        
        # Calculate frame positions
        start_frame_idx = 0
        middle_frame_idx = n_frames // 2
        end_frame_idx = n_frames - 1
        
        frames = {}
        
        # Extract the three key frames
        for frame_name, frame_idx in [('start', start_frame_idx), ('middle', middle_frame_idx), ('end', end_frame_idx)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None, {"error": f"Could not read {frame_name} frame at index {frame_idx}"}
            frames[frame_name] = frame
        
        cap.release()
        
        # Preprocess all frames
        start_processed = preprocess_frame(frames['start'], use_cuda=use_cuda)
        middle_processed = preprocess_frame(frames['middle'], use_cuda=use_cuda)
        end_processed = preprocess_frame(frames['end'], use_cuda=use_cuda)
        
        # Calculate motion between frames
        start_to_middle_motion, _, _ = get_motion_percentage(
            start_processed, middle_processed, threshold=threshold, use_cuda=use_cuda
        )
        
        middle_to_end_motion, _, _ = get_motion_percentage(
            middle_processed, end_processed, threshold=threshold, use_cuda=use_cuda
        )
        
        start_to_end_motion, _, _ = get_motion_percentage(
            start_processed, end_processed, threshold=threshold, use_cuda=use_cuda
        )
        
        # Define adaptive thresholds
        baseline_threshold = max(threshold * 0.5, 3)
        train_threshold = threshold * 1.5
        
        # Prepare diagnostics
        motion_values = {
            'start_to_middle': start_to_middle_motion,
            'middle_to_end': middle_to_end_motion,
            'start_to_end': start_to_end_motion
        }
        
        thresholds_used = {
            'base_threshold': threshold,
            'baseline_threshold': baseline_threshold,
            'train_threshold': train_threshold
        }
        
        video_info = {
            'total_frames': total_frames,
            'n_frames_used': n_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'frame_indices': {
                'start': start_frame_idx,
                'middle': middle_frame_idx,
                'end': end_frame_idx
            }
        }
        
        # Pattern analysis
        if (start_to_middle_motion >= train_threshold and 
            middle_to_end_motion >= train_threshold and 
            start_to_end_motion <= baseline_threshold):
            result = True
            pattern = "Train passage detected: high motion entering and exiting, stable baseline"
            
        elif (start_to_middle_motion <= baseline_threshold and
              middle_to_end_motion <= baseline_threshold and
              start_to_end_motion <= baseline_threshold):
            result = False  
            pattern = "No train detected: consistently low motion across all frames"
            
        else:
            result = None
            if start_to_end_motion > baseline_threshold:
                pattern = f"Anomalous: high baseline motion ({start_to_end_motion:.1f}%) suggests scene changes"
            else:
                pattern = f"Anomalous: mixed motion pattern - start->mid: {start_to_middle_motion:.1f}%, mid->end: {middle_to_end_motion:.1f}%"
        
        diagnostics = {
            'motion_values': motion_values,
            'pattern_analysis': pattern,
            'thresholds_used': thresholds_used,
            'video_info': video_info
        }
        
        return result, diagnostics
        
    except Exception as e:
        return None, {"error": f"Error processing video {video_file}: {str(e)}"}
    

import cv2
import os
import numpy as np

def debug_check_for_train(video_file: str, output_dir: str, n_frames: int = None, threshold=10, use_cuda=True):
    """
    Debug version of train detection that saves frames and diagnostic images for analysis.
    
    Args:
        video_file: Path to the video file
        output_dir: Directory to save debug images
        n_frames: Total number of frames in the video. If None, will be determined automatically
        threshold: Base threshold for motion detection (default: 10)
        use_cuda: Whether to use CUDA acceleration (default: True)
    
    Returns:
        tuple: (detection_result, debug_info_dict)
    """
    
    # try:
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return None, {"error": "Could not open video file"}
    
    # Determine frame count if not provided
    if n_frames is None:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate frame count
    if n_frames < 3:
        print(f"Error: Video has insufficient frames ({n_frames})")
        cap.release()
        return None, {"error": f"Insufficient frames ({n_frames})"}
    
    # Calculate frame positions (beginning, middle, end)
    start_frame_idx = 0
    middle_frame_idx = n_frames // 2
    end_frame_idx = n_frames - 1
    
    frames = {}
    frame_info = {}
    
    # Extract the three key frames
    for frame_name, frame_idx in [('start', start_frame_idx), ('middle', middle_frame_idx), ('end', end_frame_idx)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read {frame_name} frame at index {frame_idx}")
            cap.release()
            return None, {"error": f"Could not read {frame_name} frame"}
        
        frames[frame_name] = frame.copy()
        frame_info[frame_name] = frame_idx
        
        # Save original frame
        frame_path = os.path.join(output_dir, f"{video_name}_{frame_name}_frame_{frame_idx:06d}_original.png")
        cv2.imwrite(frame_path, frame)
    
    cap.release()

    # Preprocess all frames and save preprocessed versions
    processed_frames = {}
    for frame_name in ['start', 'middle', 'end']:
        processed = preprocess_frame(frames[frame_name], use_cuda=use_cuda)
        processed_frames[frame_name] = processed
        
        # Save preprocessed frame (convert back to uint8 for saving)
        if hasattr(processed, 'download'):  # GPU mat
            processed_cpu = processed.download()
        else:
            processed_cpu = processed
        
        # Convert to uint8 if needed
        if processed_cpu.dtype != np.uint8:
            processed_cpu = (processed_cpu * 255).astype(np.uint8) if processed_cpu.max() <= 1 else processed_cpu.astype(np.uint8)
        
        processed_path = os.path.join(output_dir, f"{video_name}_{frame_name}_frame_{frame_info[frame_name]:06d}_processed.png")
        cv2.imwrite(processed_path, processed_cpu)
    
    # Calculate motion and save difference images
    motion_results = {}
    difference_images = {}
    
    # Start to middle
    start_to_middle_motion, gpu1, gpu2 = get_motion_percentage(
        processed_frames['start'], processed_frames['middle'], threshold=threshold, use_cuda=use_cuda
    )
    motion_results['start_to_middle'] = start_to_middle_motion
    
    # Create and save difference image
    diff_img = create_difference_visualization(processed_frames['start'], processed_frames['middle'], threshold, use_cuda)
    diff_path = os.path.join(output_dir, f"{video_name}_diff_start_to_middle.png")
    cv2.imwrite(diff_path, diff_img)
    difference_images['start_to_middle'] = diff_path
    
    # Middle to end
    middle_to_end_motion, _, _ = get_motion_percentage(
        processed_frames['middle'], processed_frames['end'], threshold=threshold, use_cuda=use_cuda
    )
    motion_results['middle_to_end'] = middle_to_end_motion
    
    # Create and save difference image
    diff_img = create_difference_visualization(processed_frames['middle'], processed_frames['end'], threshold, use_cuda)
    diff_path = os.path.join(output_dir, f"{video_name}_diff_middle_to_end.png")
    cv2.imwrite(diff_path, diff_img)
    difference_images['middle_to_end'] = diff_path
    
    # Start to end (baseline)
    start_to_end_motion, _, _ = get_motion_percentage(
        processed_frames['start'], processed_frames['end'], threshold=threshold, use_cuda=use_cuda
    )
    motion_results['start_to_end'] = start_to_end_motion
    
    # Create and save difference image
    diff_img = create_difference_visualization(processed_frames['start'], processed_frames['end'], threshold, use_cuda)
    diff_path = os.path.join(output_dir, f"{video_name}_diff_start_to_end.png")
    cv2.imwrite(diff_path, diff_img)
    difference_images['start_to_end'] = diff_path
    
    # Create a summary visualization
    create_summary_visualization(frames, motion_results, video_name, output_dir, frame_info, threshold)
    
    # Define adaptive thresholds
    baseline_threshold = max(threshold * 0.5, 3)
    train_threshold = threshold * 1.5
    
    # Pattern analysis
    if (start_to_middle_motion >= train_threshold and 
        middle_to_end_motion >= train_threshold and 
        start_to_end_motion <= baseline_threshold):
        result = True
        pattern = "Train passage detected"
    elif (start_to_middle_motion <= baseline_threshold and
            middle_to_end_motion <= baseline_threshold and
            start_to_end_motion <= baseline_threshold):
        result = False  
        pattern = "No train detected"
    else:
        result = None
        pattern = "Anomalous pattern"
    
    # Compile debug info
    debug_info = {
        'video_file': video_file,
        'video_name': video_name,
        'output_dir': output_dir,
        'frame_indices': frame_info,
        'motion_values': motion_results,
        'thresholds': {
            'base_threshold': threshold,
            'baseline_threshold': baseline_threshold,
            'train_threshold': train_threshold
        },
        'pattern_analysis': pattern,
        'saved_files': {
            'original_frames': [
                os.path.join(output_dir, f"{video_name}_start_frame_{frame_info['start']:06d}_original.png"),
                os.path.join(output_dir, f"{video_name}_middle_frame_{frame_info['middle']:06d}_original.png"),
                os.path.join(output_dir, f"{video_name}_end_frame_{frame_info['end']:06d}_original.png")
            ],
            'processed_frames': [
                os.path.join(output_dir, f"{video_name}_start_frame_{frame_info['start']:06d}_processed.png"),
                os.path.join(output_dir, f"{video_name}_middle_frame_{frame_info['middle']:06d}_processed.png"),
                os.path.join(output_dir, f"{video_name}_end_frame_{frame_info['end']:06d}_processed.png")
            ],
            'difference_images': list(difference_images.values()),
            'summary': os.path.join(output_dir, f"{video_name}_summary.png")
        }
    }
    
    return result, debug_info
        
    # except Exception as e:
    #     print(f"Error processing video {video_file}: {str(e)}")
    #     return None, {"error": str(e)}


def create_difference_visualization(frame1, frame2, threshold, use_cuda=True):
    """
    Create a visualization of the difference between two frames.
    """
    try:
        # Download from GPU if needed
        if hasattr(frame1, 'download'):
            f1 = frame1.download()
        else:
            f1 = frame1.copy()
            
        if hasattr(frame2, 'download'):
            f2 = frame2.download()
        else:
            f2 = frame2.copy()
        
        # Ensure frames are the same size
        if f1.shape != f2.shape:
            f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
        
        # Calculate absolute difference
        diff = cv2.absdiff(f1, f2)
        
        # Apply threshold to create binary mask
        if len(diff.shape) == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff
        
        _, binary_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Create colored visualization
        # Green for areas below threshold, Red for areas above threshold
        colored_diff = np.zeros((diff_gray.shape[0], diff_gray.shape[1], 3), dtype=np.uint8)
        colored_diff[:, :, 1] = 255 - binary_mask  # Green channel for low motion
        colored_diff[:, :, 2] = binary_mask        # Red channel for high motion
        
        # Blend with original difference for better visualization
        diff_normalized = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
        if len(diff_normalized.shape) == 2:
            diff_normalized = cv2.cvtColor(diff_normalized, cv2.COLOR_GRAY2BGR)
        
        # Combine the visualizations
        result = cv2.addWeighted(diff_normalized, 0.7, colored_diff, 0.3, 0)
        
        return result
        
    except Exception as e:
        print(f"Error creating difference visualization: {e}")
        # Return a blank image if there's an error
        return np.zeros((480, 640, 3), dtype=np.uint8)


def create_summary_visualization(frames, motion_results, video_name, output_dir, frame_info, threshold):
    """
    Create a summary image showing all three frames and motion analysis.
    """
    try:
        # Resize frames to a consistent size for the summary
        target_height = 240
        resized_frames = []
        
        for frame_name in ['start', 'middle', 'end']:
            frame = frames[frame_name]
            aspect_ratio = frame.shape[1] / frame.shape[0]
            target_width = int(target_height * aspect_ratio)
            resized = cv2.resize(frame, (target_width, target_height))
            
            # Add frame label
            cv2.putText(resized, f"{frame_name.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(resized, f"Frame {frame_info[frame_name]}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            resized_frames.append(resized)
        
        # Create horizontal layout
        max_width = max(frame.shape[1] for frame in resized_frames)
        summary_width = max_width * 3
        summary_height = target_height + 150  # Extra space for text
        
        summary = np.zeros((summary_height, summary_width, 3), dtype=np.uint8)
        
        # Place frames
        x_offset = 0
        for i, frame in enumerate(resized_frames):
            y_end = target_height
            x_end = x_offset + frame.shape[1]
            summary[0:y_end, x_offset:x_end] = frame
            x_offset += max_width
        
        # Add motion analysis text
        text_y = target_height + 30
        motion_text = [
            f"Motion Analysis (threshold={threshold}):",
            f"Start -> Middle: {motion_results['start_to_middle']:.1f}%",
            f"Middle -> End: {motion_results['middle_to_end']:.1f}%", 
            f"Start -> End: {motion_results['start_to_end']:.1f}%"
        ]
        
        for i, text in enumerate(motion_text):
            cv2.putText(summary, text, (10, text_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save summary
        summary_path = os.path.join(output_dir, f"{video_name}_summary.png")
        cv2.imwrite(summary_path, summary)
        
    except Exception as e:
        print(f"Error creating summary visualization: {e}")


# Convenience function for batch processing
def debug_multiple_videos(video_files, base_output_dir, **kwargs):
    """
    Debug multiple video files, creating separate directories for each.
    """
    results = {}
    
    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_dir = os.path.join(base_output_dir, video_name)
        
        print(f"Processing {video_name}...")
        result, debug_info = debug_check_for_train(video_file, output_dir, **kwargs)
        results[video_file] = (result, debug_info)
        
        # Print summary
        if 'error' in debug_info:
            print(f"  ERROR: {debug_info['error']}")
        else:
            motion = debug_info['motion_values']
            print(f"  Result: {result}")
            print(f"  Motion: {motion['start_to_middle']:.1f}% | {motion['middle_to_end']:.1f}% | {motion['start_to_end']:.1f}%")
            print(f"  Files saved to: {output_dir}")
        print()
    
    return results
=======
                self.process_frame(roi)
            
            return ret, frame
    
>>>>>>> Stashed changes
