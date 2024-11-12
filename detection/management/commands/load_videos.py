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
        num_videos = len(video_paths)
        self.stdout.write(f"Processing {num_videos} videos...")
        for i, video_path in enumerate(video_paths):
            self.stdout.write(f"  Progress: {i + 1} of {num_videos} videos", ending="\r")
            video = Video(file=video_path, camera=self.params.camera)
            video.init(save=False)
            video.start = timezone.make_aware(datetime.strptime(video.filename, "VID_%Y%m%d_%H%M%S"))
            videos.append(video)
        self.stdout.write(f"\nProcessed {num_videos} videos!")
        Video.objects.bulk_create(videos)
