import os
import cv2
from django.conf import settings
from django.core.management.base import BaseCommand

from classification.deblur import create_motion_kernel, wiener_deblur
from detection.utils import ImageInterface, image_from_array
from PIL import Image


class Command(BaseCommand):
    help = "Imports a batch of videos and creates VideoHandlers."

    def add_arguments(self, parser):
        parser.add_argument('images', nargs="+")

    def handle(self, *args, **options):
        
        for img_path in options["images"]:
            sample = Image.open(img_path)
            interface = ImageInterface(sample)
            bounding_box = interface.get_bounding_box(title="Select the start and end of the motion blur")
            first_point, second_point = bounding_box

            # img = cv2.imread(img_path)
            # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # img = cv2.flip(img, 1)
            
            kernel = create_motion_kernel(first_point, second_point, kernel_size=10)

            deblur = wiener_deblur(sample, kernel, K=0.1)
            cv2.imwrite(os.path.join(settings.MEDIA_ROOT, "images", "deblur", img_path.split('/')[-1]), deblur)
