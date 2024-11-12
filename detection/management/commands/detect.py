import json
from django.core.management.base import BaseCommand

from detection.detector import Detector, ExclusionDetector
from trainspotting.utils import FFMPEG_BASE


class Command(BaseCommand):
    help = "Isolates and clips motion events from a video file"

    def add_arguments(self, parser):
        parser.add_argument('videos', nargs="+")
        parser.add_argument('-r', '--results', default="results.json", type=str)        
        parser.add_argument('-u', '--upper', default=5, type=float)
        parser.add_argument('-l', '--lower', default=1, type=float)
        parser.add_argument('-m', '--minlength', default=3, type=int)
        parser.add_argument('-x', '--exclude', default=False, action='store_true')
        parser.add_argument('--nomerge', action='store_true', default=False)

    def handle(self, *args, **options):
        video_paths = options.pop("videos")
        exclude = options.pop("exclude")
        if exclude:
            detector = ExclusionDetector(video_paths, log=self.stdout.write, **options)
        else:
            detector = Detector(video_paths, logger=self.stdout.write, **options)

        write_comma = False  # Hacky sorry
        with open(options['results'], 'w') as results:
            results.write("[")
        try:
            for data in detector.detect_loop():
                if data:
                    with open(options['results'], 'a') as results:
                        if write_comma:
                            results.write(',')
                        else:
                            write_comma = True
                        json.dump(data, results)
        finally:
            with open(options['results'], 'a') as results:
                    results.write("]")

