# File: yourapp/management/commands/stitch_blurred_text.py

import cv2
import numpy as np
from scipy.signal import correlate2d
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os

class Command(BaseCommand):
    help = 'Stitch together a clearer image from multiple blurred frames of horizontally moving text'

    def add_arguments(self, parser):
        parser.add_argument('video_path', type=str, help='Path to the video file')
        parser.add_argument('start_frame', type=int, help='Starting frame number')
        parser.add_argument('num_frames', type=int, help='Number of frames to process')

    def handle(self, *args, **options):
        video_path = options['video_path']
        start_frame = options['start_frame']
        num_frames = options['num_frames']

        if not os.path.exists(video_path):
            raise CommandError(f"Video file does not exist: {video_path}")

        try:
            stacked, enhanced = self.stitch_blurred_text(video_path, start_frame, num_frames)
            
            # Save the results
            output_dir = os.path.join(settings.MEDIA_ROOT, 'stitched_text_output')
            os.makedirs(output_dir, exist_ok=True)

            stacked_path = os.path.join(output_dir, 'stacked_image.png')
            enhanced_path = os.path.join(output_dir, 'enhanced_image.png')

            cv2.imwrite(stacked_path, stacked)
            cv2.imwrite(enhanced_path, enhanced)

            self.stdout.write(self.style.SUCCESS('Successfully stitched blurred text frames'))
            self.stdout.write(f'Stacked image saved to: {stacked_path}')
            self.stdout.write(f'Enhanced image saved to: {enhanced_path}')

        except Exception as e:
            raise CommandError(f"An error occurred: {str(e)}")

    def extract_frames(self, video_path, start_frame, num_frames):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        return frames

    def estimate_shift(self, frame1, frame2):
        correlation = correlate2d(frame1, frame2, mode='same')
        y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
        return y - frame1.shape[0]//2, x - frame1.shape[1]//2

    def align_and_stack_frames(self, frames):
        reference = frames[0]
        aligned_frames = [reference]
        
        for frame in frames[1:]:
            dy, dx = self.estimate_shift(reference, frame)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            aligned_frames.append(aligned)
        
        stacked = np.mean(aligned_frames, axis=0).astype(np.uint8)
        return stacked

    def enhance_text(self, image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def stitch_blurred_text(self, video_path, start_frame, num_frames):
        frames = self.extract_frames(video_path, start_frame, num_frames)
        stacked_image = self.align_and_stack_frames(frames)
        enhanced_image = self.enhance_text(stacked_image)
        return stacked_image, enhanced_image