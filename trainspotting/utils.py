import logging
import os
import sys
from django.conf import settings
from django.core.management.base import BaseCommand

from trainspotting.settings import FFMPEG_BASE


class VideosCommand(BaseCommand):
    default_dest = "results"

    def add_arguments(self, parser):
        parser.add_argument('videos', nargs="+")
        parser.add_argument('-d', '--destination')

    def initialize_dest(self, options):
        """Determine destination folder and create if necessary."""
        if options['dest'] is None:
            dest = os.path.join(f"{settings.BASE_DIR}", self.default_dest)
        else:
            dest = options["dest"]
        
        if not os.path.exists(dest):
            os.makedirs(dest)

        return dest


class BaseLoggingCommand(BaseCommand):
    """
    Base management command that provides configurable logging capabilities
    to be inherited by specific commands.
    """
    
    def add_arguments(self, parser):
        parser.add_argument(
            '-l',
            '--log', 
            type=str, 
            help='Path to the log file',
            default="debug.log"
        )
        parser.add_argument(
            '--log-level', 
            type=str, 
            help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
            default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        )

    def _configure_logger(self, options):
        """
        Configures logging based on command options.
        Can be overridden by subclasses for custom logging setup.
        """
        # Get log file and level from options
        log_file = os.path.join("logs", options.get('log'))
        log_level = getattr(logging, options.get('log_level', 'INFO'))

        # Configure base logging
        log_config = {
            'level': log_level,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }

        log_config['handlers'] = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
        
        logging.basicConfig(**log_config)
        
        return logging.getLogger(self.__class__.__name__)

    def handle(self, *args, **options):
        """
        Subclasses should override this method and call super()._configure_logger(options)
        """
        raise NotImplementedError("Subclasses must implement handle method")
    

class BaseCameraCommand(BaseLoggingCommand):
    """Base management command that implements a get_camera method."""

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument("-c", "--camera", required=True)

    def get_camera(self, options):
        from detection.models import Camera

        try:
            return Camera.objects.get(name=options['camera'])
        except Camera.DoesNotExist:
            self.stderr.write(f"{options['camera']} does not exist.")
            confirmation = input("Would you like to create it? Y/n").lower().strip() or 'y'
            if confirmation:
                address_confirm = False
                while not address_confirm:
                    address = input(f"Please enter the address of {options['camera']} camera: ")
                    address_confirm = (input(f"Is '{address}' correct? Y/n").lower().strip() or 'y') != 'n'

                return Camera.objects.create(name=options['camera'], address=address)
            else:
                sys.exit(1)


def concat_clips(paths):
    """Concatenates a series of clips together into the first clip location."""
    for path in paths:
        os.system(f'echo "file {path}" >> merge.txt')
    os.system(f"{FFMPEG_BASE} -f concat -safe 0 -i merge.txt -c copy merge.mp4")
    os.system(f"mv merge.mp4 {path[0]}")
    os.system(f"rm -f merge.txt {' '.join(paths[1:])}")
    