from argparse import BooleanOptionalAction
from detection.tasks import import_videos
from trainspotting.utils import BaseCameraCommand


class Command(BaseCameraCommand):
    help = "Imports a batch of videos and creates VideoHandlers."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('videos', nargs="+")
        parser.add_argument('--in-place', action="store_true")

    def handle(self, *args, **options):
        logger = self._configure_logger(options)
        camera = self.get_camera(options)
        video_paths = options.pop("videos")

        import_videos(video_paths, camera, logger=logger, in_place=options["in_place"])