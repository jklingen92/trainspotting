from detection.models import Detection, VideoBatch
from detection.tasks import detect_clips
from trainspotting.utils import BaseLoggingCommand
from django.core.management.base import CommandError


class Command(BaseLoggingCommand):
    help = "Detects clips in a batch of Videos."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('import', type=int)
        parser.add_argument('--view', default="", type=str)
        parser.add_argument('-d', '--detection', type=int, default=None)

    def handle(self, *args, **options):
        logger = self._configure_logger(options)

        if not options['detection'] and not options['view']:
            CommandError("You must specify either a view or a detection instance.")

        try:
            video_batch = VideoBatch.objects.get(id=options["import"])
        except VideoBatch.DoesNotExist as e:
            raise CommandError(e)

        if options['detection']:
            try:
                detection = Detection.objects.get(id=options["detection"])
            except Detection.DoesNotExist as e:
                raise CommandError(e)
            
            detect_clips(video_batch.handlers.all(), detection=detection, logger=logger)

        else:

            detect_clips(video_batch.handlers.all(), view=options["view"], logger=logger)

        