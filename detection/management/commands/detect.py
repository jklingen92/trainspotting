import json
from django.core.management.base import BaseCommand

from detection.detector import Detector, ExclusionDetector
from trainspotting.utils import FFMPEG_BASE


class Command(BaseCommand):
    help = "Isolates and clips motion events from a video file"

    def add_arguments(self, parser):
        parser.add_argument('videos', nargs="+")
        parser.add_argument('-c', '--camera', default="camera", type=str)
        parser.add_argument('-u', '--upper', default=5, type=float)
        parser.add_argument('-l', '--lower', default=1, type=float)
        parser.add_argument('-m', '--minlength', default=5, type=int)
        parser.add_argument('-x', '--exclude', default=False, action='store_true')
        parser.add_argument('-f', '--fake', default=False, action='store_true')

    def handle(self, *args, **options):
        video_paths = options.pop("videos")
        exclude = options.pop("exclude")
        fake = options.pop("fake")
        if exclude:
            detector = ExclusionDetector(video_paths, log=self.stdout.write, **options)
        else:
            detector = Detector(video_paths, logger=self.stdout.write, **options)

        clips = detector.detect_loop(save=not fake)
        