import cv2
from django.core.management.base import BaseCommand
from matplotlib import pyplot as plt
import numpy as np


class Command(BaseCommand):
    help = "Show a frame from a video"

    def add_arguments(self, parser):
        parser.add_argument('video')
        parser.add_argument('frame', type=int)

    def handle(self, *args, **options):
        cap = cv2.VideoCapture(options["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, options['frame'])
        _, img = cap.read()
        plt.imshow(img)
        plt.show()