# recorder_app/management/commands/record_video.py

import os
import sys
import time
import cv2
import numpy as np
from django.core.management.base import BaseCommand

from detection.pipeline import GStreamerPipeline


class Command(BaseCommand):
    help = 'Record video using OpenCV and GStreamer with custom parameters'

    def add_arguments(self, parser):
        parser.add_argument(
            '--duration',
            type=int,
            default=10,
            help='Recording duration in seconds (default: 10)'
        )
        parser.add_argument(
            '--output',
            type=str,
            required=True,
            help='Output file path (e.g., /path/to/output.mp4)'
        )
        parser.add_argument(
            '--exposure',
            type=int,
            default=0,
            help='Camera exposure value in microseconds (default: auto)'
        )
        parser.add_argument(
            '--sensor-mode',
            type=int,
            default=0,
            help='Camera sensor mode (0: 4K@30fps, 1: 4K@30fps, 2: 1080p@30fps)'
        )
        parser.add_argument(
            '--bitrate',
            type=int,
            default=50000,
            help='Encoding bitrate in Kbps (default: 50000)'
        )
        parser.add_argument(
            '--framerate',
            type=int,
            default=30,
            help='Camera framerate (default: 30)'
        )

    def handle(self, *args, **options):
        duration = options['duration']
        output_path = options['output']
        exposure = options['exposure']
        sensor_mode = options['sensor_mode']
        bitrate = options['bitrate']
        framerate = options['framerate']
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Determine resolution based on sensor mode
        if sensor_mode in [0, 1]:
            width, height = 3840, 2160  # 4K
        elif sensor_mode == 2:
            width, height = 1920, 1080  # 1080p
        else:
            self.stderr.write(self.style.ERROR(f"Invalid sensor mode: {sensor_mode}"))
            return
        
        # Build GStreamer pipelines for capture and output
        pipeline = GStreamerPipeline(
            sensor_mode=sensor_mode,
            framerate=framerate, 
        )

        cap = pipeline.open_capture(
            exposure=exposure,
        )
        
        out = pipeline.open_output(
            bitrate=bitrate, 
            output_path=output_path
        )
 
        self.stdout.write(f"Started recording to {output_path}")
        self.stdout.write(f"Resolution: {width}x{height}, Duration: {duration}s")
        
        start_time = time.time()
        frames_captured = 0
        last_progress_update = 0
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    self.stderr.write(self.style.WARNING("Failed to read frame"))
                    time.sleep(0.01)  # Brief pause to avoid CPU hogging on failure
                    continue
                
                frames_captured += 1
                
                # Here you can process the frame for recognition
                # Example: frame = self._process_frame(frame)
                
                # Write frame to output
                out.write(frame)
                
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Only update progress every 0.5 seconds to reduce terminal output
                if current_time - last_progress_update >= 0.5:
                    fps = frames_captured / elapsed if elapsed > 0 else 0
                    progress = min(100, int(elapsed / duration * 100))
                    sys.stdout.write(f"\rRecording: {progress}% complete | Time: {int(elapsed)}s/{duration}s | FPS: {fps:.1f} | Frames: {frames_captured}  ")
                    sys.stdout.flush()
                    last_progress_update = current_time
                    
        except KeyboardInterrupt:
            self.stdout.write("Recording stopped by user")
        finally:
            cap.release()
            out.release()
            
            # Calculate actual FPS
            elapsed = time.time() - start_time
            fps = frames_captured / elapsed if elapsed > 0 else 0
            
            self.stdout.write(self.style.SUCCESS(
                f"Recording completed. Duration: {elapsed:.2f}s, "
                f"Frames: {frames_captured}, Avg FPS: {fps:.2f}"
            ))
    
    def _process_frame(self, frame):
        """
        Process the frame for recognition or other operations
        This is where you would add your recognition code
        """
        # Example processing (you can replace with your actual processing)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # return processed
        
        return frame  # Return original frame if no processing needed