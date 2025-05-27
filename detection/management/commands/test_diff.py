# motion_detector/management/commands/capture_motion.py

import cv2
import os
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from detection.motion_capture import MotionCapture

class Command(BaseCommand):
    help = 'Captures and analyzes motion from a video file or camera'

    def add_arguments(self, parser):
        # Required arguments
        parser.add_argument('source', type=str, 
                            help='Video file path or camera index (0 for default camera)')
        
        # Optional arguments
        parser.add_argument('--threshold', type=int, default=25,
                            help='Threshold for motion detection (default: 25)')
        parser.add_argument('--min-area', type=float, default=0.05,
                            help='Minimum percentage of frame that must change (default: 0.05 or 5%%)')
        parser.add_argument('--skip-frames', type=int, default=5,
                            help='Process every Nth frame for motion detection (default: 5)')
        parser.add_argument('--min-motion-seconds', type=float, default=1.0,
                            help='Minimum motion duration in seconds (default: 1.0)')
        parser.add_argument('--min-stillness-seconds', type=float, default=1.0,
                            help='Minimum stillness duration in seconds (default: 1.0)')
        parser.add_argument('--resize-width', type=int, default=480,
                            help='Width to resize frames for processing (default: 480)')
        parser.add_argument('--cuda', action='store_true',
                            help='Enable CUDA acceleration')
        parser.add_argument('--start-frame', type=int, default=0,
                            help='Start processing from this frame number (default: 0)')

    def handle(self, *args, **options):
        # Parse source - camera index or file path
        source = options['source']
        if source.isdigit():
            source = int(source)
        elif not os.path.exists(source):
            raise CommandError(f"Video file not found: {source}")
        
        try:
            # Initialize MotionCapture
            motion_cap = MotionCapture(
                source=source,
                skip_frames=options['skip_frames'],
                threshold=options['threshold'],
                min_area_percentage=options['min_area'],
                min_motion_seconds=options['min_motion_seconds'],
                min_stillness_seconds=options['min_stillness_seconds'],
                resize_width=options['resize_width'],
                use_cuda=options['cuda'],
            )
            
            if not motion_cap.isOpened():
                raise CommandError(f"Could not open video source: {source}")
            
            motion_cap.set(cv2.CAP_PROP_POS_FRAMES, options['start_frame'])
            motion_cap.frame_count = options['start_frame']
            self.stdout.write(f"Starting motion capture from: {source}")
            
            start_time = timezone.now()
            self.stdout.write(f"Start time: {start_time}")

            # Main processing loop
            while True:
                success, _ = motion_cap.read()
                if not success:
                    break
                
                print(f"Processing frame {motion_cap.frame_count} - Motion: {int(motion_cap.motion_percentage * 100):2d}% {motion_cap.motion_detected} - {motion_cap.consecutive_motion_frames} / {motion_cap.consecutive_still_frames}")

               
            end_time = timezone.now()
            frames_processed = motion_cap.frame_count - options['start_frame']
            time_elapsed = (end_time - start_time).total_seconds()
            self.stdout.write(self.style.SUCCESS("Motion capture complete"))
            self.stdout.write(f"End time: {end_time}")
            self.stdout.write(f"Processed {frames_processed} frames in {time_elapsed:.2f} seconds | Effective FPS: {frames_processed / time_elapsed:.2f}")

        except Exception as e:
            raise CommandError(f"Error during motion capture: {str(e)}")
        
        finally:
            # Clean up
            if 'motion_cap' in locals():
                motion_cap.release()
            