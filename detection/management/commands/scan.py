import os
import cv2
from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Imports a batch of videos and creates VideoHandlers."

    def add_arguments(self, parser):
        parser.add_argument('videos', nargs="+")

    def handle(self, *args, **options):
        n_videos = len(options["videos"])

        for i, video_path in enumerate(options["videos"]):
            print(f"Scanning {i} of {n_videos}...\r")
            cap = cv2.VideoCapture(video_path)
            _, img1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 24 * 60 * 1)
            _, img2 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 24 * 60 * 2)
            _, img3 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 24 * 60 * 3)
            _, img4 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 24 * 60 * 4)
            _, img5 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 24 * 60 * 5)
            _, img6 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 24 * 60 * 6)
            _, img7 = cap.read()

            video_name = video_path.split('/')[-1].split(".")[0]
            try:
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, "images", f"{video_name}-{1}.png"), img1)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, "images", f"{video_name}-{2}.png"), img2)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, "images", f"{video_name}-{3}.png"), img3)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, "images", f"{video_name}-{4}.png"), img4)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, "images", f"{video_name}-{5}.png"), img5)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, "images", f"{video_name}-{6}.png"), img6)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, "images", f"{video_name}-{7}.png"), img7)
            except cv2.error:
                continue
