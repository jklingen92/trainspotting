from detection.detector import Detector, ExclusionDetector
from detection.models import DetectTask, ImportTask
from detection.utils import image_from_array
from trainspotting.utils import BaseLoggingCommand
from django.core.management.base import CommandError


class Command(BaseLoggingCommand):
    help = "Runs a detect algorithm on an ImportTask id, creating Clips out of the Videos."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('detect', type=int)

    def handle(self, *args, **options):
        logger = self._configure_logger(options)

        try:
            detect_task = DetectTask.objects.get(id=options["detect"])
        except DetectTask.DoesNotExist as e:
            raise CommandError(e)
        
        clips = detect_task.clips.all()
        logger.info(f"Extracting {clips.count()} clips to {detect_task.clip_destination}...")
        for clip in clips:
            logger.info(f"  Extracting {clip.outfile}...")
            clip.extract()
        logger.info(f"Extracted {clips.count()} clips to {detect_task.clip_destination}!")

