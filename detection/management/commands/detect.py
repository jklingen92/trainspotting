from typing import Any
from django.core.management.base import BaseCommand, CommandParser
from detection.detector import Detector
from detection.models import Video


class Command(BaseCommand):
    help = "Runs detect loop on all videos with files attached"

    def add_arguments(self, parser):
        parser.add_argument('-l', '--log')

    def handle(self, *args, **options):
        videos = Video.objects.exclude(file=None)

        def logger(msg, **kwargs):
            if kwargs.get("ending", None) == "\r":
                return
            else:
                with open(options["log"], "a") as f:
                    f.write(f"{msg}\n")
        
        detector = Detector(videos, logger=logger)
        try:
            stubs = detector.detect_loop()
            logger(f"Created {len(stubs)} stubs!")

        except Exception as e:
            logger(e)