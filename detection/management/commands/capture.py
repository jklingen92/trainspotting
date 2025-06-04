import traceback
import cv2
import logging
from django.core.management.base import BaseCommand
import os
from django.utils import timezone

from detection.motion_capture import MotionCapture
from detection.pipeline import GStreamerPipeline

logger = logging.getLogger("detection")

class Command(BaseCommand):
    help = 'Test video capture with specified parameters'

    def add_arguments(self, parser):
        parser.add_argument('--output-dir', type=str, default=None, help='Path to store videos motion')
        parser.add_argument('--duration', type=int, default=0, help='Duration of the capture in seconds (0 for continuous capture)')
        
        parser.add_argument('--sensor-mode', type=int, default=0, help='Sensor mode (0 or 1 for 4K, 2 for 1080p)')
        parser.add_argument('--exposure', type=int, default=450000, help='Exposure time in microseconds')
        parser.add_argument('--framerate', type=int, default=30, help='Frames per second')
        parser.add_argument('--bitrate', type=int, default=50000, help='Bitrate in kbps')
        
        parser.add_argument('--min-clip-length', type=int, default=3, help='Minimum length in seconds for a saved clip')
        parser.add_argument('--max-frame-buffer', type=int, default=90, help='Maximum number of frames to buffer before writing to disk')
        

    def handle(self, *args, **options):
        output_dir = options['output_dir']
        duration = options['duration']

        sensor_mode = options['sensor_mode']
        exposure = options['exposure']
        framerate = options['framerate']
        bitrate = options['bitrate']
        
        min_clip_length = options['min_clip_length']
        max_frame_buffer = options['max_frame_buffer']

        # Generate output file name if not provided
        if output_dir is None:
            timestamp = timezone.localtime().strftime("%Y%m%d_%H%M%S")
            output_dir = f"camera_capture_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)

        # Log the parameters
        self.stdout.write(self.style.SUCCESS('Starting capture with the following parameters:'))
        logger.info(f'Output File: {output_dir}')
        logger.info(f'Duration: {duration} seconds')

        logger.info(f'Sensor Mode: {sensor_mode}')
        logger.info(f'Exposure: {exposure} nanoseconds')
        logger.info(f'Framerate: {framerate} fps')
        logger.info(f'Bitrate: {bitrate} kbps')

        logger.info(f'Minimum Clip Length: {min_clip_length} seconds')
        logger.info(f'Maximum Frame Buffer: {max_frame_buffer} frames')

        # Create GStreamer pipeline for camera capture
        pipeline = GStreamerPipeline(
            sensor_mode=sensor_mode,
            capture_class=MotionCapture,
        )

        cap = pipeline.open_capture(
            exposure=exposure,
        )

        logger.info(f"Recording settings: exposure={exposure}ns, {pipeline.width}x{pipeline.height} @ {framerate}fps")

        start_time = current_time = timezone.localtime()
        frame_buffer_full = False
        out = current_clip = None
        frame_buffer = []

        try:
            while True:
                current_time = timezone.localtime()
                if duration > 0 and (current_time - start_time).total_seconds() >= duration:
                    logger.info(f"Capture duration reached {duration} seconds, stopping capture.")
                    break

                timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')

                ret, frame = cap.read()
                self.stdout.write(f"Processing frame at {timestamp} - Motion: {int(cap.motion_percentage * 100):2d}% {cap.motion_detected} - {cap.consecutive_motion_frames} / {cap.consecutive_still_frames}", ending='\r')
                if not ret:
                    logger.info(f"Failed to grab frame")
                    break

                # Draw timestamp on the frame
                cv2.putText(frame, timestamp, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
                

                if cap.motion_detected and out is None:
                    current_clip = f"clip_{timezone.localtime().strftime('%Y%m%d_%H%M%S')}.mp4"
                    out = pipeline.open_output(os.path.join(output_dir, current_clip), bitrate=bitrate, framerate=framerate)
                    motion_start_time = current_time
                    logger.info(f"Motion detected, starting new clip: {current_clip}")
                
                elif not cap.motion_detected and out is not None:
                    clip_duration = (current_time - motion_start_time).total_seconds()
                    
                    # Check if the clip is long enough
                    if clip_duration < min_clip_length:
                        os.remove(os.path.join(output_dir, current_clip))
                        logger.info(f"Motion ended but clip too short ({clip_duration:.2f}s), deleting {current_clip}")
                    else:
                        logger.info(f"Motion ended, saved clip {current_clip} ({clip_duration:.2f}s)")

                    logger.info(f"Releasing output for clip: {current_clip}")
                    out.release()
                    
                    out = current_clip = None
                    frame_buffer.clear()  # Clear buffer when stopping recording
                    frame_buffer_full = False

                if frame_buffer_full or len(frame_buffer) == max_frame_buffer: 
                    frame_buffer_full = True
                    if out:
                        out.write(frame_buffer.pop(0))
                    else:
                        frame_buffer.pop(0)

                frame_buffer.append(frame)

            # Clean up
            if out is not None:
                out.release()
                logger.info(f"Final clip saved: {current_clip}")

        except Exception as e:
            logger.info(f"ERROR: {str(e)}")
            logger.info("Traceback:")
            traceback.print_exc()
            return False