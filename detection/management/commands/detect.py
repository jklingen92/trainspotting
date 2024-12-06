from detection.detector import Detector, ExclusionDetector
from detection.models import DetectTask, ImportTask
from detection.utils import image_from_array
from trainspotting.utils import BaseLoggingCommand
from django.core.management.base import CommandError


class Command(BaseLoggingCommand):
    help = "Runs a detect algorithm on an ImportTask id, creating Clips out of the Videos."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('import', type=int)
        parser.add_argument('--view', default="", type=str)

    def handle(self, *args, **options):
        logger = self._configure_logger(options)

        try:
            import_task = ImportTask.objects.get(id=options["import"])
        except ImportTask.DoesNotExist as e:
            raise CommandError(e)

        first_processing_task = import_task.processing_tasks.first()
        first_frame = first_processing_task.read()
        first_processing_task.release()

        sample_name = f"{first_processing_task.video.filename}_sample.png"
        detect_task, created = DetectTask.objects.get_or_create(
            import_task=import_task,
            view=options["view"],
            defaults={
                "sample": image_from_array(sample_name, first_frame.image)
            }
        )
        
        detect_task.detect = detect_task.get_bounding_box("Select a detection area or leave blank for the full area")
        detect_task.exclude = detect_task.get_bounding_box("Select an exclusion area or leave blank for no exclusion")
        detect_task.save()

        if detect_task.exclude is None:
            detector = Detector(detect_task, logger=logger)
        else:
            detector = ExclusionDetector(detect_task, logger=logger)

        try:
            detector.detect_loop()
            logger.info(f"Created {detect_task.clips.count()} clips")

        except Exception as e:
            logger.error(e)
            raise e
        