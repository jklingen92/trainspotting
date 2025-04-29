from datetime import datetime
import time
import traceback
import uuid
from django.core.management.base import BaseCommand
import os
import cv2
from django.conf import settings

from detection.difference import detect_large_motion
from detection.pipeline import build_pipeline

class Command(BaseCommand):
    help = 'Capture video with specified parameters'

    def add_arguments(self, parser):
        parser.add_argument('--output_dir', type=str, default=None, help='Path to store captured motion')
        parser.add_argument('--duration', type=int, default=0, help='Duration of the capture in seconds (0 for continuous capture)')
        parser.add_argument('--exposure', type=int, default=450000, help='Exposure time in microseconds')
        parser.add_argument('--auto_gain', type=bool, default=True, help='Enable or disable auto gain')
        parser.add_argument('--resolution', type=str, default='3840x2160', help='Resolution in WIDTHxHEIGHT format')
        parser.add_argument('--framerate', type=int, default=30, help='Frames per second')
        parser.add_argument('--bitrate', type=int, default=50000, help='Bitrate in kbps')
        parser.add_argument('--max_file_size', type=int, default=0, help='Maximum file size in GB (0 for unlimited)')
        parser.add_argument('--buffer', type=int, help='Seconds to add onto beginning and end of clip')
        parser.add_argument('--min_clip_length', type=int, default=3, help='Minimum length in seconds for a saved clip')
        parser.add_argument('--test', action="store_true", help='Record continuously and log motion detection to a file.')
        

    def handle(self, *args, **options):
        output_dir = options['output_dir']
        duration = options['duration']
        exposure = options['exposure']
        auto_gain = options['auto_gain']
        resolution = options['resolution']
        framerate = options['framerate']
        bitrate = options['bitrate']
        max_file_size = options['max_file_size']
        buffer = options['buffer']
        min_clip_length = options['min_clip_length']
        test = options['test']

        # Parse resolution
        try:
            width, height = map(int, resolution.split('x'))
        except ValueError:
            self.stderr.write(self.style.ERROR('Invalid resolution format. Use WIDTHxHEIGHT.'))
            return

        # Generate output file name if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"camera_capture_{timestamp}"
        
        buffer_frames = buffer * framerate

        os.makedirs(output_dir, exist_ok=True)

        # Log the parameters
        self.stdout.write(self.style.SUCCESS('Starting capture with the following parameters:'))
        self.stdout.write(f'Output File: {output_dir}')
        self.stdout.write(f'Duration: {duration} seconds')
        self.stdout.write(f'Exposure: {exposure} microseconds')
        self.stdout.write(f'Auto Gain: {auto_gain}')
        self.stdout.write(f'Resolution: {width}x{height}')
        self.stdout.write(f'Framerate: {framerate} fps')
        self.stdout.write(f'Bitrate: {bitrate} kbps')
        self.stdout.write(f'Max File Size: {max_file_size} GB')

        # Placeholder for actual capture logic
        self.stdout.write(self.style.SUCCESS('Capture logic not implemented yet.'))

        # Create GStreamer pipeline for camera capture
        pipeline_str = build_pipeline(
            camera_id=0,
            resolution=resolution,
            framerate=framerate,
            exposure=exposure,
            bitrate=bitrate,
            auto_gain=auto_gain,
            preview=True
        )

        self.stdout.write(f"Camera pipeline: {pipeline_str}")
        
        cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            self.stdout.write("Failed to open camera!")
            return False
        
        self.stdout.write(f"Camera opened successfully")

        # Allow camera to adjust for the first second
        settle_time = time.time()
        while (time.time() - settle_time) < 1.0:
            ret, _ = cap.read()  # Discard frames while settling

        # Start recording
        total_frame_count = 0

        # Function to create a new video writer
        def create_writer(output_path, width, height):
            gst_out = (
                "appsrc ! "
                "video/x-raw, format=BGR ! "
                "videoconvert ! "
                "video/x-raw, format=I420 ! "
                f"x264enc speed-preset=ultrafast tune=zerolatency bitrate={bitrate} ! "
                "h264parse ! "
                "mp4mux ! "
                f"filesink location={output_path}"
            )
            
            writer = cv2.VideoWriter(
                gst_out, 
                cv2.CAP_GSTREAMER, 
                0,
                float(framerate),
                (width, height)
            )
            
            if not writer.isOpened():
                self.stdout.write(f"Failed to create video writer for {output_path}!")
                return None
                
            self.stdout.write(f"Recording to {output_path}")
            return writer
        
        # Read first frame to get dimensions
        ret, test_frame = cap.read()
        if not ret:
            self.stdout.write("Failed to read initial frame")
            cap.release()
            return False
            
        width = test_frame.shape[1]
        height = test_frame.shape[0]
        
        frame_count = 0  # Reset frame count for current file
        bytes_per_frame_estimate = (width * height * 3) / (8 * 1024 * 1024 * 1024)  # Rough estimate in GB
        expected_frames_per_gb = 1.0 / bytes_per_frame_estimate if bytes_per_frame_estimate > 0 else 0
        self.stdout.write(f"Recording settings: exposure={exposure}ns, auto_gain={auto_gain}, {width}x{height} @ {framerate}fps")
        self.stdout.write(f"Estimated size per frame: {bytes_per_frame_estimate:.4f} GB")

        frame_buffer = []
        end_buffer = buffer_frames
        still_frames = 15
        recording = True if test else False
        temp_file = None
        out = None
        motion_start_time = None

        if test:
            if duration > 0:
                self.stdout.write(f"Recording for {duration} seconds. Motion events will be logged to {output_dir}/motion.log")
            else:
                self.stdout.write(f"Recording continuously until Ctrl+C is pressed. Motion events will be logged to {output_dir}/motion.log")
        if duration > 0:
            self.stdout.write(f"Capturing for {duration} seconds. Motion events will be saved in {output_dir}")
        else:
            self.stdout.write(f"Capturing continuously until Ctrl+C is pressed. Motion events will be saved in {output_dir}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.stdout.write(f"Failed to grab frame")
                    break
                
                frame_buffer.append(frame.copy())
                if len(frame_buffer) > buffer_frames:
                    frame_buffer.pop(0)
                frame_count += 1
                total_frame_count += 1


                # Only start motion detection when we have enough frames
                if len(frame_buffer) < 2:
                    continue

                # Check for motion
                motion_detected, _, _ = detect_large_motion(
                    frame_buffer[-2], frame_buffer[-1], threshold=15
                )

                if motion_detected:
                    if not recording:
                        # Start of motion - create temp file
                        recording = True
                        temp_id = str(uuid.uuid4())
                        temp_file = os.path.join(output_dir, f"temp_{temp_id}.mp4")
                        out = create_writer(temp_file, *resolution)
                        motion_start_time = time.time()
                        
                        # Write buffer frames first (pre-motion context)
                        for buffered_frame in frame_buffer[:-1]:  # Exclude current frame
                            out.write(buffered_frame)
                            
                    # Write current frame
                    if out:
                        out.write(frame)

                    if test:
                        # Log motion event to 'motion.log'
                        with open(os.path.join(output_dir, "motion.log"), "a") as log_file:
                            log_file.write(f"Motion detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Frame: {frame_count}\n")
                    
                else:
                    
                    elapsed = time.time() - motion_start_time if motion_start_time else 0

                    # No motion detected
                    if recording:

                        if elapsed >=  min_clip_length:
                            if end_buffer == 0:
                                if test:
                                    # Log motion end to 'motion.log'
                                    with open(os.path.join(output_dir, "motion.log"), "a") as log_file:
                                        log_file.write(f"Motion ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Duration: {elapsed:.1f} seconds\n")
                                else:
                                    # Motion sustained - finalize this clip
                                    final_file = os.path.join(output_dir, f"motion_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                                    self.stdout.write(f"Motion ended after {elapsed:.1f} seconds - saving clip")
                                    
                                    recording = False
                                    if out:
                                        out.release()
                                        out = None
                                        temp_file = None

                                    
                                    # Rename temp file to final name
                                    os.rename(temp_file, final_file)
                                    motion_start_time = None
                                    end_buffer = buffer_frames
                                    still_frames = 15
                                    continue
                            else:
                                end_buffer -= 1
                        
                        else:
                            # Motion ended but not sustained - wait for still frames
                            if still_frames == 0:
                                # remove clip
                                if test:
                                    # Log motion end to 'motion.log'
                                    with open(os.path.join(output_dir, "motion.log"), "a") as log_file:
                                        log_file.write(f"Motion ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Duration: {elapsed:.1f} seconds - not sustained\n")
                                else:
                                    self.stdout.write(f"Motion ended after {elapsed:.1f} seconds - not sustained")
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                    
                                    recording = False
                                    if out:
                                        out.release()
                                        out = None
                                        temp_file = None

                                    motion_start_time = None
                                    end_buffer = buffer_frames
                                    still_frames = 15
                                    continue
                            else:
                                still_frames -= 1

                        # Write current frame unless we're stopping
                        if out:
                            out.write(frame)    
                        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Clean up
            if out:
                out.release()

            # Delete any lingering temp file
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            self.stdout.write(f"ERROR: {str(e)}")
            self.stdout.write("Traceback:")
            traceback.print_exc()
            return False