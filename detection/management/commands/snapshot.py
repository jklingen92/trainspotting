import os
import cv2
from django.core.management.base import BaseCommand, CommandError

from detection.capture_pipeline import CapturePipeline

class Command(BaseCommand):
    help = 'Takes a snapshot from camera using OpenCV with GStreamer'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            default='snapshot.png',
            help='Output file path'
        )
        parser.add_argument(
            '--sensor-mode',
            type=int,
            default=0,
            help='Sensor mode (0: 4K@30fps, 1: 4K@30fps, 2: 1080p@30fps)'
        )
        parser.add_argument(
            '--exposure',
            default='450000',
            help='Exposure time (in ns)'
        )
        parser.add_argument(
            '--warmup-frames',
            type=int,
            default=15,
            help='Number of frames to capture for auto-adjustment before final snapshot'
        )

    def handle(self, *args, **options):
        output_path = options['output']
        sensor_mode = options['sensor_mode']
        exposure = options['exposure']
        warmup_frames = options['warmup_frames']

        try:
            pipeline = CapturePipeline(sensor_mode=sensor_mode, exposure=exposure, warmup_frames=warmup_frames)
            
            # Take the actual snapshot
            self.stdout.write("Taking snapshot...")
            ret, frame = pipeline.cap.read()
            
            if not ret or frame is None:
                raise CommandError("Failed to grab frame for snapshot")
                
            self.stdout.write(f"Frame captured successfully. Shape: {frame.shape}, Type: {frame.dtype}")
            cv2.imwrite(output_path, frame)

            # Release the pipeline
            pipeline.release()
            
            if os.path.exists(output_path):
                self.stdout.write(self.style.SUCCESS(f"Snapshot saved to {output_path}"))
            else:
                raise CommandError("Output file does not exist after saving attempt")
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
            # Try to release the camera if it's still open
            try:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
            except:
                pass
            raise CommandError(f"Error taking snapshot: {str(e)}")