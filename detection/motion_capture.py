import cv2
import numpy as np
from threading import Lock

from django.utils import timezone

def preprocess_frame(frame, resize_width=480, canny_low=50, canny_high=150):
    """Preprocess the frame for motion detection"""
    if frame is None:
        return None
        
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    
    resized_gpu = cv2.cuda.resize(gpu_frame, (resize_width, int(frame.shape[0] * (resize_width / frame.shape[1]))))
    gpu_gray = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gpu_gray.download(), (5, 5), 0)

    gpu_blur = cv2.cuda_GpuMat()
    gpu_blur.upload(blurred_frame)

    # Perform Canny edge detection on GPU
    canny_detector = cv2.cuda.createCannyEdgeDetector(low_thresh=canny_low, high_thresh=canny_high)
    return canny_detector.detect(gpu_blur)
    # edges = gpu_edges.download()
    # return edges, gpu_edges


def debug_gpumat(gpu_mat, name):
    """Debug helper to check GpuMat properties"""
    print(f"{name}:")
    print(f"  Empty: {gpu_mat.empty()}")
    print(f"  Size: {gpu_mat.size()}")  # height, width
    print(f"  Type: {gpu_mat.type()}")
    print(f"  Channels: {gpu_mat.channels()}")
    print(f"  Depth: {gpu_mat.depth()}")


def get_motion_percentage_cuda(current_edges_gpu, previous_edges_gpu, max_flow_magnitude=50.0):
    """
    Calculate motion using optical flow on edge images. Returns value between 0 and 1.
    
    Args:
        current_edges_gpu: Current frame edges (GpuMat)
        previous_edges_gpu: Previous frame edges (GpuMat)
        max_flow_magnitude: Maximum expected flow magnitude for normalization (pixels per frame)
        
    Returns:
        float: Motion intensity between 0.0 (no motion) and 1.0 (maximum motion)
    """
    if previous_edges_gpu.empty() or current_edges_gpu.empty():
        return 0.0
    
    assert previous_edges_gpu.size() == current_edges_gpu.size(), "Edge images must be the same size for optical flow calculation"


    try:
        # Create Farneback optical flow calculator
        flow_calculator = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=3,           # Number of pyramid levels
            pyrScale=0.5,      # Pyramid scale factor
            fastPyramids=True,
            winSize=15,         # Window size
            numIters=3,       # Iterations per pyramid level
            polyN=5,           # Neighborhood size for polynomial approximation
            polySigma=1.2,      # Standard deviation for Gaussian kernel
            flags=0
        )
        
        # Calculate optical flow

        flow = cv2.cuda_GpuMat(previous_edges_gpu.size(), cv2.CV_32FC2)  # Flow output
        flow_calculator.calc(previous_edges_gpu, current_edges_gpu, flow)
        
        # Split flow into x and y components
        flow_parts = cv2.cuda.split(flow)
        flow_x = flow_parts[0]  # Horizontal flow
        flow_y = flow_parts[1]  # Vertical flow
        # Calculate magnitude of flow vectors
        magnitude = cv2.cuda_GpuMat()
        cv2.cuda.magnitude(flow_x, flow_y, magnitude)
        
        # Calculate mean flow magnitude across all pixels
        total_flow = cv2.cuda.sum(magnitude)[0]  # Sum of all flow magnitudes
        total_pixels = magnitude.rows * magnitude.cols
        
        if total_pixels == 0:
            return 0.0
            
        mean_flow_magnitude = total_flow / total_pixels
        
        # Normalize to 0-1 range using max_flow_magnitude
        # Use a smooth saturation function instead of hard clipping
        normalized_motion = mean_flow_magnitude / max_flow_magnitude
        
        # Apply sigmoid-like saturation to handle values > 1.0 gracefully
        # This ensures very high motion doesn't just clip to 1.0
        if normalized_motion > 1.0:
            normalized_motion = 1.0 - (1.0 / (1.0 + normalized_motion - 1.0))
        
        return min(max(normalized_motion, 0.0), 1.0)  # Clamp to [0, 1]
        
    except Exception as e:
        print(f"Optical flow calculation failed: {e}")
        return 0.0


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
    """
    
    def __init__(self, *args, skip_frames=5, threshold=10, min_area_start=0.05, min_area_end=0.05,
                 resize_width=360, fps=30, min_motion_seconds=1.0, 
                 min_stillness_seconds=1.0, learning_frames=30, **kwargs):
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
        self.learning_frames = learning_frames
        
        # Initialize motion detection variables
        self.frame_count = 0
        self.background = cv2.cuda_GpuMat()  # Background frame for motion detection
        self.motion_detected = False
        self.consecutive_motion_frames = 0
        self.consecutive_still_frames = 0
        self.motion_start_time = None
        self.motion_percentage = 0.0
        self.last_motion_time = None
        
        # Lock for thread safety
        self.lock = Lock()
    
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
                
                if self.frame_count % self.skip_frames == 0:
                    # Preprocess the frame 
                    gpu_current = preprocess_frame(frame, resize_width=self.resize_width)
                    motion_percentage = get_motion_percentage_cuda(
                        self.background,
                        gpu_current
                    )

                    # Update background (simple approach - could use more sophisticated background subtraction)
                    # alpha = 0.05  # Learning rate
                    
                    # GPU-based background update
                    # if self.background.empty():
                    #     self.background = gpu_current.clone()
                    # else:
                    #     cv2.cuda.addWeighted(self.background, 1 - alpha, gpu_current, alpha, 0)
                    self.background = gpu_current
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
        processed = preprocess_frame(frames[frame_name])
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
    start_to_middle_motion = get_motion_percentage_cuda(
        processed_frames['start'], processed_frames['middle']
    )
    motion_results['start_to_middle'] = start_to_middle_motion
    
    # Create and save difference image
    diff_img = create_difference_visualization(processed_frames['start'], processed_frames['middle'], threshold, use_cuda)
    diff_path = os.path.join(output_dir, f"{video_name}_diff_start_to_middle.png")
    cv2.imwrite(diff_path, diff_img)
    difference_images['start_to_middle'] = diff_path
    
    # Middle to end
    middle_to_end_motion = get_motion_percentage_cuda(
        processed_frames['middle'], processed_frames['end']
    )
    motion_results['middle_to_end'] = middle_to_end_motion
    
    # Create and save difference image
    diff_img = create_difference_visualization(processed_frames['middle'], processed_frames['end'], threshold, use_cuda)
    diff_path = os.path.join(output_dir, f"{video_name}_diff_middle_to_end.png")
    cv2.imwrite(diff_path, diff_img)
    difference_images['middle_to_end'] = diff_path
    
    # Start to end (baseline)
    start_to_end_motion = get_motion_percentage_cuda(
        processed_frames['start'], processed_frames['end']
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