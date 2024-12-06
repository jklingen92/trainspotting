import os
from django.conf import settings

from detection.models import ImportTask, VideoProcessingTask
from trainspotting.utils import BaseCameraCommand


class Command(BaseCameraCommand):
    help = "Imports a batch of videos."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('videos', nargs="+")

    def handle(self, *args, **options):
        logger = self._configure_logger(options)
        camera = self.get_camera(options)
        import_task = ImportTask.objects.create(camera=camera)
        video_paths = options.pop("videos")
        num_videos = len(video_paths)
        staging_location = os.path.join(settings.MEDIA_ROOT, 'raw', camera.name)
        logger.info(f"Importing {num_videos} videos to {staging_location}:")
        try:
            for i, video_path in enumerate(video_paths):
                filename = video_path.split('/')[-1]
                logger.info(f"  Importing {filename} ({i + 1} of {num_videos})")
                os.system(f"rsync {video_path} {staging_location}/")
                task = VideoProcessingTask.objects.create(import_task=import_task, file=os.path.join('raw', camera.name, filename))
                task.init()
        except Exception as e:
            logger.error(e)
        finally:
            logger.info(f"Successfully created {import_task.processing_tasks.count()} processing_tasks (ImportTask #{import_task.id})")

