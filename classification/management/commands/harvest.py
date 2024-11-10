import os
import cv2
from django.core.management.base import BaseCommand
from random import randint
from trainspotting.utils import Video, VideosCommand


class Command(VideosCommand):
    help = "Harvests random frames from several train videos to build a set of training data."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('-b', '--buffer', default=7, type=int)
        parser.add_argument('-f', '--frames', default=10, type=int)

    def handle(self, *args, **options):
        dest = self.initialize_dest(options)
        n_videos = len(options['videos'])
        total_frames = 0
        for i, video_path in enumerate(options["videos"]):
            self.stdout.write(f"Harvesting video {i + 1} of {n_videos}: {video_path}...")
            video = Video(video_path)
            minutes = video.duration / 60000
            frames_to_capture = round(options['frames'] * minutes) + 1
            
            # Get lower and upper frame bounds based on buffer
            video.seek_milli(options['buffer'] * 1000)
            start_frame = video.pos_frame
            
            video.seek_milli(video.duration - (options['buffer'] * 1000))
            end_frame = video.pos_frame

            harvested = 0
            while harvested < frames_to_capture:
                frame = randint(start_frame, end_frame)
                video.seek_frame(frame)
                s, img = video.cap.read()
                if not s:
                    raise Exception("Failed to read video.")
                
                filename = f"{video.pos_datetime.strftime('%F_%H%M%S_%f')}.jpg"
                outfile = os.path.join(dest, filename)
                if os.path.exists(outfile):  # Don't resave an existing frame
                    continue
                self.stdout.write(f"  Saving frame {frame} ({harvested + 1} of {frames_to_capture}): {filename}...")
                cv2.imwrite(outfile, img)
                harvested += 1 
                total_frames += 1
        
        self.stdout.write(f"Harvested {total_frames} frames from {n_videos} videos to {dest}")

