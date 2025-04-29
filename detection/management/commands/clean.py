from django.core.management.base import BaseCommand
from detection.models import Video


class Command(BaseCommand):
    help = "Removes file from Video instances"


    def handle(self, *args, **options):
        Video.objects.update(file=None)