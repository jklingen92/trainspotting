from datetime import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone

from detection.models import Video


class Command(BaseCommand):
    help = "Creates video objects for a batch of videos."

    def add_arguments(self, parser):
        parser.add_argument('videos', nargs="+")

    def handle(self, *args, **options):
        video_paths = options.pop("videos")
        videos = []
        for video_path in video_paths:
            video = Video(file=video_path, camera=self.params.camera)
            video.init(save=False)
            video.start = timezone.make_aware(datetime.strptime(video.filename, "VID_%Y%m%d_%H%M%S"))
            videos.append(video)
        Video.objects.bulk_create(videos)
