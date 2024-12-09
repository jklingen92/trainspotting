from detection.models import Detection
from detection.tasks import extract_clips
from trainspotting.utils import BaseLoggingCommand
from django.core.management.base import CommandError


class Command(BaseLoggingCommand):
    help = "Extracts clips from a detection operation."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('detect', type=int)

    def handle(self, *args, **options):
        logger = self._configure_logger(options)

        try:
            detection = Detection.objects.get(id=options["detect"])
        except Detection.DoesNotExist as e:
            raise CommandError(e)
        
        extract_clips(detection, logger=logger)