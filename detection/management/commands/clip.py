from django.core.management.base import BaseCommand
from detection.models import ClipStub


class Command(BaseCommand):
    help = "Make clips based on clip stubs"

    def add_arguments(self, parser):
        parser.add_argument('-d', 'destination', type=str)

    def handle(self, *args, **options):
        destination = options['destination']
        stubs = ClipStub.objects.filter(clip=None).exclude(video__file=None).order_by("video__start", "start")
        for stub in stubs:
            if stub.merge_to:
                stub.clip_and_merge(stub.merge_to)
            else:
                stub.clip(destination)