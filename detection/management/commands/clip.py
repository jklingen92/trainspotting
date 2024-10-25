from functools import cached_property
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from django.conf import settings
from django.utils import timezone
from django.core.management.base import BaseCommand
from numpy import Infinity
from datetime import datetime, timedelta

from detection.utils import FFMPEG_BASE


class Command(BaseCommand):
    help = "Simple wrapper for FFMPEG"

    def add_arguments(self, parser):
        parser.add_argument('video')
        parser.add_argument('outfile')
        parser.add_argument('-s', '--start', default=0)
        parser.add_argument('-e', '--end', default=None)

    def handle(self, *args, **options):
        end_str = ""
        if options['end'] is not None:
            end_str = f"-to '{options['end']}ms'"
        os.system(f"{FFMPEG_BASE} -ss '{options['start']}ms' {end_str} -i {options['video']} {options['outfile']}")
                    