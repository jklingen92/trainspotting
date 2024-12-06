from django.core.management.base import BaseCommand
from detection.models import Clip
from trainspotting.utils import BaseLoggingCommand


class Command(BaseLoggingCommand):
    help = "Make clips based on clip stubs"

    def add_arguments(self, parser):
        super().add_arguments(parser)
        

    def handle(self, *args, **options):
        destination = options['destination']
        stubs = Clip.objects.filter(clip=None).exclude(video__file=None).order_by("video__start", "start")
        for stub in stubs:
            if stub.merge_to:
                stub.clip_and_merge(stub.merge_to)
            else:
                stub.clip(destination)