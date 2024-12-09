from detection.models import VideoBatch
from detection.tasks import detect_clips
from trainspotting.utils import BaseLoggingCommand
from django.core.management.base import CommandError


class Command(BaseLoggingCommand):
    help = "Detects clips in a batch of Videos."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('import', type=int)
        parser.add_argument('--view', default="", type=str)

    def handle(self, *args, **options):
        logger = self._configure_logger(options)

        try:
            video_batch = VideoBatch.objects.get(id=options["import"])
        except VideoBatch.DoesNotExist as e:
            raise CommandError(e)

        detect_clips(video_batch.handlers.all(), view=options["view"], logger=logger)

        