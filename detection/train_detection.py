#!/usr/bin/env python3

import gi
import cv2
import numpy as np
import threading
import time
import os
import subprocess
import json
from collections import deque
from datetime import datetime

gi.require_version('Gst', '1.0')
# gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib

import logging

logger = logging.getLogger('detection')

# Check for CUDA support
CUDA_AVAILABLE = False
try:
    # Check if OpenCV was compiled with CUDA support
    CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if CUDA_AVAILABLE:
        logger.info(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}")
except AttributeError:
    logger.warning("OpenCV not compiled with CUDA support")
except Exception as e:
    logger.warning(f"CUDA check failed: {e}")

# Check for TensorRT availability
TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    logger.info("TensorRT available")
except ImportError:
    logger.info("TensorRT not available")

# CUDA memory management
if CUDA_AVAILABLE:
    # Set CUDA memory pool to avoid fragmentation
    cv2.cuda.setDevice(0)
    cv2.cuda.resetDevice()


class TrainDetectionSystem:
    def __init__(self, **kwargs):
        # CSI Camera configuration
        self.sensor_mode = kwargs.get('sensor_mode', 0)
        self.exposure = kwargs.get('exposure', 450000)
        
        # Set resolution based on sensor mode
        if self.sensor_mode in [0, 1]:
            # 4K modes
            self.video_width = kwargs.get('video_width', 3840)
            self.video_height = kwargs.get('video_height', 2160)
        else:
            # 1080p mode
            self.video_width = kwargs.get('video_width', 1920)
            self.video_height = kwargs.get('video_height', 1080)
        
        # Recording configuration
        self.output_dir = kwargs.get('output_dir', './recordings')
        self.buffer_seconds = kwargs.get('buffer_seconds', 1)
        self.fps = kwargs.get('fps', 30)
        
        # SSH configuration
        self.remote_host = kwargs.get('remote_host', '')
        self.remote_user = kwargs.get('remote_user', '')
        self.remote_path = kwargs.get('remote_path', '')
        self.ssh_key = kwargs.get('ssh_key', '')
        
        # Motion detection parameters
        self.motion_threshold = kwargs.get('motion_threshold', 5000)
        self.min_recording_duration = kwargs.get('min_recording_duration', 5.0)
        
        # Analysis parameters
        self.enable_car_counting = kwargs.get('enable_car_counting', True)
        self.enable_speed_calculation = kwargs.get('enable_speed_calculation', True)
        self.enable_placard_detection = kwargs.get('enable_placard_detection', True)
        self.speed_calibration_factor = kwargs.get('speed_calibration_factor', 0.1)
        
        # Video parameters
        self.video_quality = kwargs.get('video_quality', 'medium')
        
        # CUDA acceleration settings
        self.use_cuda = kwargs.get('use_cuda', CUDA_AVAILABLE)
        self.gpu_memory_fraction = kwargs.get('gpu_memory_fraction', 0.8)
        
        # Logging
        self.verbose = kwargs.get('verbose', False)
        
        # Tracking variables
        self.frame_buffer = deque(maxlen=self.buffer_seconds * self.fps * 2)
        
        # Add these lines after your existing GPU matrix setup
        if self.use_cuda and CUDA_AVAILABLE:
            self.gpu_prev_gray = cv2.cuda_GpuMat()
            self.optical_flow = None  # Will be created on first use
        
        
        self.is_recording = False
        self.recording_start_time = None
        self.last_motion_time = None
        self.current_recording_path = None
        
        # Train analysis
        self.train_cars = []
        self.speed_measurements = []
        self.placard_detections = []
        
        # Threading
        self.recording_thread = None
        self.upload_queue = []
        self.shutdown_event = threading.Event()
        
        # Initialize directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize GStreamer
        Gst.init(None)
        self.setup_pipeline()
        
    def setup_pipeline(self):
        """Setup GStreamer pipeline for CSI camera with hardware acceleration"""
        # Quality settings with updated bitrates
        quality_settings = {
            'low': {'bitrate': 10000},
            'medium': {'bitrate': 20000},
            'high': {'bitrate': 50000}
        }
        
        bitrate = quality_settings.get(self.video_quality, quality_settings['medium'])['bitrate']
        
        # CSI camera pipeline with hardware acceleration and exposure control
        pipeline_str = f"""
        nvarguscamerasrc sensor-mode={self.sensor_mode} exposuretimerange="{self.exposure} {self.exposure}" ! 
        video/x-raw(memory:NVMM),width={self.video_width},height={self.video_height},framerate={self.fps}/1,format=NV12 ! 
        nvvidconv ! 
        video/x-raw,format=BGRx ! 
        videoconvert !
        video/x-raw,format=BGR !
        appsink name=sink emit-signals=true max-buffers=1 drop=true
        """
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsink = self.pipeline.get_by_name('sink')
            self.appsink.connect('new-sample', self.on_new_sample)
            logger.info(f"CSI camera pipeline created - Sensor mode: {self.sensor_mode}, "
                       f"Resolution: {self.video_width}x{self.video_height}, Exposure: {self.exposure}")
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise
        
    def on_new_sample(self, sink):
        """Callback for new video frames"""
        if self.shutdown_event.is_set():
            return Gst.FlowReturn.EOS
            
        sample = sink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Extract frame info
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            
            # Convert to numpy array
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                frame = frame_data.reshape((height, width, 3))[:,:,:3]
                buffer.unmap(map_info)
                
                self.process_frame(frame)
                
        return Gst.FlowReturn.OK
    
    def process_frame(self, frame):
        """Process each frame for motion detection and analysis"""
        timestamp = time.time()
        
        # Add frame to buffer with timestamp
        self.frame_buffer.append((frame.copy(), timestamp))
        
        # Motion detection
        motion_detected = self.detect_motion(frame)
        
        if motion_detected:
            self.last_motion_time = timestamp
            if not self.is_recording:
                self.start_recording()
        
        # Check if we should stop recording
        if (self.is_recording and 
            self.last_motion_time and 
            timestamp - self.last_motion_time > self.buffer_seconds and
            timestamp - self.recording_start_time > self.min_recording_duration):
            self.stop_recording()
        
        # Train analysis if recording
        if self.is_recording:
            self.analyze_train(frame, timestamp)
    
    def detect_motion(self, frame):
        """Detect motion using CUDA-accelerated optical flow"""
        if not hasattr(self, 'prev_gray'):
            if self.use_cuda and CUDA_AVAILABLE:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                self.gpu_prev_gray = cv2.cuda_GpuMat()
                cv2.cuda.cvtColor(gpu_frame, self.gpu_prev_gray, cv2.COLOR_BGR2GRAY)
            else:
                self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False
        
        if self.use_cuda and CUDA_AVAILABLE:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_gray = cv2.cuda_GpuMat()
            
            gpu_frame.upload(frame)
            cv2.cuda.cvtColor(gpu_frame, gpu_gray, cv2.COLOR_BGR2GRAY)
            
            # Create Farneback optical flow
            if not hasattr(self, 'optical_flow'):
                self.optical_flow = cv2.cuda_FarnebackOpticalFlow.create()
            
            gpu_flow = cv2.cuda_GpuMat()
            self.optical_flow.calc(self.gpu_prev_gray, gpu_gray, gpu_flow)
            
            # Split flow into x and y components
            gpu_flow_x = cv2.cuda_GpuMat()
            gpu_flow_y = cv2.cuda_GpuMat()
            cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
            
            # Calculate magnitude
            gpu_mag = cv2.cuda_GpuMat()
            gpu_ang = cv2.cuda_GpuMat()
            cv2.cuda.cartToPolar(gpu_flow_x, gpu_flow_y, gpu_mag, gpu_ang)
            
            # Threshold motion (adjust 2.0 as needed)
            gpu_thresh = cv2.cuda_GpuMat()
            cv2.cuda.threshold(gpu_mag, gpu_thresh, 2.0, 255, cv2.THRESH_BINARY)
            
            motion_pixels = cv2.cuda.countNonZero(gpu_thresh)
            
            # Update previous frame
            gpu_gray.copyTo(self.gpu_prev_gray)
            
        else:
            # CPU fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if not hasattr(self, 'flow_params'):
                self.flow_params = dict(
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
            
            flow = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, None, None, **self.flow_params)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Threshold motion
            _, thresh = cv2.threshold(mag, 2.0, 255, cv2.THRESH_BINARY)
            motion_pixels = cv2.countNonZero(thresh.astype(np.uint8))
            
            self.prev_gray = gray
        
        if self.verbose and motion_pixels > 0:
            logger.debug(f"Optical flow motion pixels: {motion_pixels}")
        
        return motion_pixels > self.motion_threshold

    def analyze_train(self, frame, timestamp):
        """Analyze train for car counting, speed, and placard detection"""
        if self.enable_car_counting:
            self._count_cars(frame, timestamp)
        
        if self.enable_speed_calculation:
            self._calculate_speed(frame, timestamp)
        
        if self.enable_placard_detection:
            self._detect_placards(frame, timestamp)
    
    def _count_cars(self, frame, timestamp):
        """Count train cars using CUDA-accelerated contour detection"""
        if self.use_cuda and CUDA_AVAILABLE:
            # GPU-accelerated edge detection
            gpu_frame = cv2.cuda_GpuMat()
            gpu_gray = cv2.cuda_GpuMat()
            gpu_edges = cv2.cuda_GpuMat()
            
            gpu_frame.upload(frame)
            cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2GRAY, gpu_gray)
            
            # Apply Gaussian blur on GPU for noise reduction
            gpu_blurred = cv2.cuda_GpuMat()
            cv2.cuda.GaussianBlur(gpu_gray, gpu_blurred, (5, 5), 0)
            
            # Canny edge detection on GPU
            cv2.cuda.Canny(gpu_blurred, gpu_edges, 50, 150)
            
            # Download result to CPU for contour detection (OpenCV limitation)
            edges = gpu_edges.download()
        else:
            # CPU fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
        
        # Contour detection (CPU only in OpenCV)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (approximate train car dimensions)
        car_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Basic filtering for train car-like objects
            if (area > 5000 and area < 50000 and 
                aspect_ratio > 1.5 and aspect_ratio < 6.0):
                car_contours.append((x, y, w, h, timestamp))
        
        self.train_cars.extend(car_contours)
        
        if self.verbose and car_contours:
            logger.debug(f"Detected {len(car_contours)} potential train cars")
    
    def _calculate_speed(self, frame, timestamp):
        """Calculate train speed"""
        if len(self.train_cars) >= 2:
            # Calculate movement between frames
            latest_cars = [car for car in self.train_cars if timestamp - car[4] < 1.0]
            if len(latest_cars) >= 2:
                # Simple speed estimation based on position change
                dx = abs(latest_cars[-1][0] - latest_cars[-2][0])
                dt = latest_cars[-1][4] - latest_cars[-2][4]
                if dt > 0:
                    pixel_speed = dx / dt
                    # Convert to real-world speed using calibration factor
                    speed_mph = pixel_speed * self.speed_calibration_factor
                    self.speed_measurements.append(speed_mph)
                    
                    if self.verbose:
                        logger.debug(f"Calculated speed: {speed_mph:.2f} mph")
    
    def _detect_placards(self, frame, timestamp):
        """Detect UN hazard placards"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for common placard colors
        color_ranges = {
            'orange': ([10, 100, 100], [25, 255, 255]),
            'red': ([0, 100, 100], [10, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255])
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area for placard
                    x, y, w, h = cv2.boundingRect(contour)
                    # Store placard detection
                    self.placard_detections.append({
                        'timestamp': timestamp,
                        'bbox': (x, y, w, h),
                        'color': color_name,
                        'area': area
                    })
                    
                    if self.verbose:
                        logger.debug(f"Detected {color_name} placard at ({x}, {y})")
    
    def start_recording(self):
        """Start recording with buffer frames"""
        self.is_recording = True
        self.recording_start_time = time.time()
        
        # Generate filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_recording_path = os.path.join(
            self.output_dir, f"train_{timestamp_str}.mp4"
        )
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_video)
        self.recording_thread.start()
        
        logger.info(f"Started recording: {self.current_recording_path}")
    
    def stop_recording(self):
        """Stop recording and add post-buffer frames"""
        if self.is_recording:
            self.is_recording = False
            
            # Wait for recording thread to finish
            if self.recording_thread:
                self.recording_thread.join()
            
            # Generate metadata
            self._save_metadata()
            
            # Queue for upload
            if self.remote_host:
                self.upload_queue.append(self.current_recording_path)
                threading.Thread(target=self._upload_video, 
                               args=(self.current_recording_path,)).start()
            
            logger.info(f"Stopped recording: {self.current_recording_path}")
            
            # Clear analysis data for next recording
            self.train_cars.clear()
            self.speed_measurements.clear()
            self.placard_detections.clear()
    
    def _record_video(self):
        """Record video with pre and post buffer frames using hardware encoding"""
        if not self.current_recording_path:
            return
        
        # Use hardware encoder on Jetson if available
        if self.use_cuda and 'nvenc' in cv2.getBuildInformation():
            # Hardware-accelerated encoding
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            logger.info("Using hardware-accelerated H.264 encoding")
        else:
            # Software encoding fallback
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            logger.info("Using software encoding")
        
        writer = cv2.VideoWriter(
            self.current_recording_path, fourcc, self.fps, 
            (self.video_width, self.video_height)
        )
        
        try:
            # Write pre-buffer frames
            buffer_frames = list(self.frame_buffer)
            recording_start_idx = max(0, len(buffer_frames) - self.buffer_seconds * self.fps)
            
            for frame, _ in buffer_frames[recording_start_idx:]:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            # Continue writing frames while recording
            while self.is_recording and not self.shutdown_event.is_set():
                if self.frame_buffer:
                    frame, _ = self.frame_buffer[-1]
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)
                time.sleep(1.0 / self.fps)
            
            # Write post-buffer frames
            post_buffer_count = 0
            max_post_buffer = self.buffer_seconds * self.fps
            
            while (post_buffer_count < max_post_buffer and self.frame_buffer and 
                   not self.shutdown_event.is_set()):
                frame, _ = self.frame_buffer[-1]
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                post_buffer_count += 1
                time.sleep(1.0 / self.fps)
                
        except Exception as e:
            logger.error(f"Error during video recording: {e}")
        finally:
            writer.release()
    
    def _save_metadata(self):
        """Save recording metadata as JSON"""
        if not self.current_recording_path:
            return
        
        # Calculate unique car count (deduplicate by position)
        unique_cars = set()
        for car in self.train_cars:
            # Group cars by approximate position
            pos_key = (car[0] // 50, car[1] // 50)  # 50-pixel grid
            unique_cars.add(pos_key)
        
        metadata = {
            'recording_path': self.current_recording_path,
            'start_time': self.recording_start_time,
            'end_time': time.time(),
            'duration': time.time() - self.recording_start_time,
            'car_count': len(unique_cars),
            'average_speed_mph': float(np.mean(self.speed_measurements)) if self.speed_measurements else 0,
            'max_speed_mph': float(np.max(self.speed_measurements)) if self.speed_measurements else 0,
            'placard_detections': self.placard_detections,
            'total_detections': len(self.train_cars),
            'camera_config': {
                'sensor_mode': self.sensor_mode,
                'exposure': self.exposure,
                'resolution': f"{self.video_width}x{self.video_height}",
                'fps': self.fps
            },
            'analysis_enabled': {
                'car_counting': self.enable_car_counting,
                'speed_calculation': self.enable_speed_calculation,
                'placard_detection': self.enable_placard_detection
            },
            'configuration': {
                'motion_threshold': self.motion_threshold,
                'buffer_seconds': self.buffer_seconds,
                'speed_calibration_factor': self.speed_calibration_factor,
                'cuda_enabled': self.use_cuda
            }
        }
        
        metadata_path = self.current_recording_path.replace('.mp4', '_metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _upload_video(self, video_path):
        """Upload video and metadata to remote server via SSH"""
        if not all([self.remote_host, self.remote_user, self.remote_path]):
            logger.warning("SSH configuration incomplete, skipping upload")
            return
        
        try:
            # Build SSH command
            ssh_cmd_base = ['scp']
            if self.ssh_key:
                ssh_cmd_base.extend(['-i', self.ssh_key])
            
            # Upload video file
            video_cmd = ssh_cmd_base + [
                video_path, 
                f"{self.remote_user}@{self.remote_host}:{self.remote_path}/"
            ]
            subprocess.run(video_cmd, check=True, timeout=300)
            
            # Upload metadata file
            metadata_path = video_path.replace('.mp4', '_metadata.json')
            if os.path.exists(metadata_path):
                metadata_cmd = ssh_cmd_base + [
                    metadata_path,
                    f"{self.remote_user}@{self.remote_host}:{self.remote_path}/"
                ]
                subprocess.run(metadata_cmd, check=True, timeout=60)
            
            logger.info(f"Successfully uploaded {os.path.basename(video_path)}")
            
            # Optionally remove local files after successful upload
            # os.remove(video_path)
            # if os.path.exists(metadata_path):
            #     os.remove(metadata_path)
            
        except subprocess.TimeoutExpired:
            logger.error(f"Upload timeout for {video_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Upload failed for {video_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down detection system...")
        self.shutdown_event.set()
        
        if self.is_recording:
            self.stop_recording()
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
    
    def run(self):
        """Start the detection system"""
        logger.info("Starting train detection system...")
        logger.info(f"CSI Camera - Sensor mode: {self.sensor_mode}, "
                   f"Resolution: {self.video_width}x{self.video_height}, "
                   f"Exposure: {self.exposure}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Motion threshold: {self.motion_threshold}")
        logger.info(f"CUDA acceleration: {'enabled' if self.use_cuda else 'disabled'}")
        
        # Start the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            # Run main loop
            loop = GLib.MainLoop()
            
            # Handle shutdown gracefully
            def signal_handler():
                logger.info("Received shutdown signal")
                self.shutdown()
                loop.quit()
            
            # Register signal handler for graceful shutdown
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, 2, signal_handler)  # SIGINT
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, 15, signal_handler)  # SIGTERM
            
            loop.run()
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            self.shutdown()