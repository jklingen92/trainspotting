import cv2
import os
import numpy as np

from json import load
from detection.detector import Detector
from trainspotting.utils import VideosCommand, Video, FFMPEG_BASE


class Command(VideosCommand):
    help = "Isolates and clips motion events from a video file"

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('-u', '--upper', default=5, type=float)
        parser.add_argument('-l', '--lower', default=1, type=float)
        parser.add_argument('-m', '--minlength', default=3, type=int)
        parser.add_argument('-f', '--fake', default=False, action='store_true')
        parser.add_argument('--nomerge', action='store_true', default=False)

    def handle(self, *args, **options):
        video_paths = options.pop("videos")
        detector = Detector(video_paths, logger=self.stdout.write, **options)
        detector.process_videos()
        print(detector.clips)
        # detector.clip_videos()



