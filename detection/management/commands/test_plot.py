from django.core.management.base import BaseCommand
from matplotlib import pyplot as plt
import numpy as np


class Command(BaseCommand):
    help = "Simple wrapper for FFMPEG"

    def handle(self, *args, **options):
        img_data = np.random.rand(100, 100)
        plt.imshow(img_data)
        plt.show()