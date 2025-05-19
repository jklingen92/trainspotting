from datetime import datetime
import sys
import time
import traceback
import uuid
from django.core.management.base import BaseCommand
import os
import cv2
from django.conf import settings

from detection.difference import detect_large_motion
from detection.pipeline import GStreamerPipeline

class Command(BaseCommand):
    help = 'Test video capture with specified parameters'

    def add_arguments(self, parser):
        parser.add_argument('--output_dir', type=str, default=None, help='Path to store videos motion')
        parser.add_argument('--max_file_size', type=int, default=0, help='Maximum file size in GB (0 for unlimited)')
        parser.add_argument('--duration', type=int, default=3600, help='Duration of the capture in seconds (0 for continuous capture)')
        
        parser.add_argument('--sensor_mode', default=0, help='Sensor mode (0 or 1 for 4K, 2 for 1080p)')
        parser.add_argument('--exposure', type=int, default=450000, help='Exposure time in microseconds')
        parser.add_argument('--framerate', type=int, default=30, help='Frames per second')
        parser.add_argument('--bitrate', type=int, default=50000, help='Bitrate in kbps')
        
        parser.add_argument('--min_clip_length', type=int, default=3, help='Minimum length in seconds for a saved clip')
        

    def handle(self, *args, **options):
        output_dir = options['output_dir']
        max_file_size = options['max_file_size']
        duration = options['duration']

        sensor_mode = options['sensor_mode']
        exposure = options['exposure']
        framerate = options['framerate']
        bitrate = options['bitrate']
        
        min_clip_length = options['min_clip_length']

        # Generate output file name if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"camera_capture_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)

        # Log the parameters
        self.stdout.write(self.style.SUCCESS('Starting capture with the following parameters:'))
        self.stdout.write(f'Output File: {output_dir}')
        self.stdout.write(f'Max File Size: {max_file_size} GB')
        self.stdout.write(f'Duration: {duration} seconds')

        self.stdout.write(f'Sensor Mode: {sensor_mode}')
        self.stdout.write(f'Exposure: {exposure} nanoseconds')
        self.stdout.write(f'Framerate: {framerate} fps')
        self.stdout.write(f'Bitrate: {bitrate} kbps')

        self.stdout.write(f'Minimum Clip Length: {min_clip_length} seconds')

        # Create GStreamer pipeline for camera capture
        pipeline = GStreamerPipeline(
            sensor_mode=sensor_mode,
        )

        cap = pipeline.open_capture(
            exposure=exposure,
        )


        frame_count = 0  # Reset frame count for current file
        bytes_per_frame_estimate = (pipeline.width * pipeline.height * 3) / (8 * 1024 * 1024 * 1024)  # Rough estimate in GB
        expected_frames_per_gb = 1.0 / bytes_per_frame_estimate if bytes_per_frame_estimate > 0 else 0
        self.stdout.write(f"Recording settings: exposure={exposure}ns, {pipeline.width}x{pipeline.height} @ {framerate}fps")
        self.stdout.write(f"Estimated size per frame: {bytes_per_frame_estimate:.4f} GB")

        clip_number = 0
        start_time = time.time()
        with open(os.path.join(output_dir, "motion.log"), "a") as log_file:
            log_file.write(f"Started recording at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        last_progress_update = 0
        frame_count = 0
        total_frame_count = 0

        motion_frames = 0
        still_frames = 0
        previous_frame = None
        

        out = pipeline.open_output(os.path.join(output_dir, f"capture_0.mp4"))

        self.stdout.write(f"Recording for {duration} seconds. Motion events will be logged to {output_dir}/motion.log")

        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    self.stdout.write(f"Failed to grab frame")
                    break
                
                frame_count += 1
                total_frame_count += 1

                current_time = time.time()
                elapsed = current_time - start_time
                
                # Only update progress every 0.5 seconds to reduce terminal output
                if current_time - last_progress_update >= 0.5:
                    fps = total_frame_count / elapsed if elapsed > 0 else 0
                    progress = min(100, int(elapsed / duration * 100))
                    sys.stdout.write(f"\rRecording: {progress}% complete | Time: {int(elapsed)}s/{duration}s | FPS: {fps:.1f} | Frames: {total_frame_count}  ")
                    sys.stdout.flush()
                    last_progress_update = current_time


                # Only start motion detection when we have enough frames
                if previous_frame is None:
                    previous_frame = frame.copy()
                    continue
                    

                # Check for motion
                motion_detected, _, _ = detect_large_motion(
                    previous_frame, frame, threshold=15
                )

                if motion_detected:
                    still_frames = 0
                    if motion_frames == 0:
                        motion_frames += 1
                        with open(os.path.join(output_dir, "motion.log"), "a") as log_file:
                            log_file.write(f"Motion begun at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} after {still_frames} still frames\n")
                    else:
                        motion_frames += 1
                else:
                    if motion_frames == 0:
                        still_frames += 1
                    else:
                        still_frames = 1
                        with open(os.path.join(output_dir, "motion.log"), "a") as log_file:
                            log_file.write(f"Motion ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} after {motion_frames} frames\n")
                        motion_frames = 0

                # Write current frame unless we're stopping
                if out:
                    out.write(frame)    
                        
                previous_frame = frame.copy()

                if frame_count >= expected_frames_per_gb * max_file_size:
                    self.stdout.write(f"Max file size reached, creating new file")
                    out.release()
                    clip_number += 1
                    out = pipeline.open_output(os.path.join(output_dir, f"capture_{clip_number}.mp4"))
                    frame_count = 0
                
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break


            with open(os.path.join(output_dir, "motion.log"), "a") as log_file:
                log_file.write(f"Ended recording at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            # Clean up
            pipeline.release()

        except Exception as e:
            self.stdout.write(f"ERROR: {str(e)}")
            self.stdout.write("Traceback:")
            traceback.print_exc()
            return False