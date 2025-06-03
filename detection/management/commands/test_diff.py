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
        parser.add_argument('--threshold', type=int, help='Threshold for motion detection (default: 25)')
        parser.add_argument('--min-area-start', type=float, help='Minimum percentage of frame that must change to start motion (default: 0.05 or 5%%)')
        parser.add_argument('--min-area-end', type=float, help='Minimum percentage of frame that must change to end motion(default: 0.25 or 25%%)')
        parser.add_argument('--skip-frames', type=int, help='Process every Nth frame for motion detection (default: 5)')
        parser.add_argument('--min-motion-seconds', type=float, help='Minimum motion duration in seconds (default: 1.0)')
        parser.add_argument('--min-stillness-seconds', type=float, help='Minimum stillness duration in seconds (default: 1.0)')
        parser.add_argument('--resize-width', type=int, help='Width to resize frames for processing (default: 480)')
        parser.add_argument('--cuda', action='store_true', help='Enable CUDA acceleration')
        parser.add_argument('--start-frame', type=int, default=0,
                            help='Start processing from this frame number (default: 0)')

    def handle(self, *args, **options):
        # Parse source - camera index or file path
        source = options['source']
        if source.isdigit():
            source = int(source)
        elif not os.path.exists(source):
            raise CommandError(f"Video file not found: {source}")
        
        kwargs = {}
        for key in ['threshold', 'min_area_start', 'min_area_end', 'skip_frames', 
                  'min_motion_seconds', 'min_stillness_seconds', 'resize_width']:
            if options[key] is not None:
                kwargs[key] = options[key]
        if options['cuda']:
            kwargs['use_cuda'] = options['cuda']


        try:
            # Initialize MotionCapture
            motion_cap = MotionCapture(
                source,
                **kwargs,
            )
            
            if not motion_cap.isOpened():
                raise CommandError(f"Could not open video source: {source}")
            
            motion_cap.set(cv2.CAP_PROP_POS_FRAMES, options['start_frame'])
            motion_cap.frame_count = options['start_frame']
            self.stdout.write(f"Starting motion capture from: {source}")
            
            start_time = timezone.localtime()
            self.stdout.write(f"Start time: {start_time}")

            recording = False
            # Main processing loop
            while True:
                success, frame = motion_cap.read()
                if not success:
                    break
                
                self.stdout.write(f"Processing frame {motion_cap.frame_count} - Motion: {int(motion_cap.motion_percentage * 100):2d}% {motion_cap.motion_detected} - {motion_cap.consecutive_motion_frames} / {motion_cap.consecutive_still_frames}", ending='\r')
                
                if motion_cap.motion_detected and not recording:
                    recording = True
                    self.stdout.write(f"Motion detected at frame {motion_cap.frame_count}")
                    cv2.imwrite(f"motion_frame_{motion_cap.frame_count}.png", frame)
                if not motion_cap.motion_detected and recording:
                    recording = False
                    self.stdout.write(f"Motion ended at frame {motion_cap.frame_count}")
                    cv2.imwrite(f"still_frame_{motion_cap.frame_count}.png", frame)
               
            end_time = timezone.localtime()
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
            