
from functools import cached_property
import os

from datetime import datetime, timedelta
import cv2
from django.conf import settings
from django.utils import timezone
from django.core.management.base import BaseCommand


FFMPEG_BASE = "ffmpeg -hide_banner -loglevel repeat+info"


class Video:
    def __init__(self, path) -> None:
        self._cap = None
        self.path = path
        self.end = datetime.fromtimestamp(os.path.getctime(path))
        self.start = self.end - timedelta(milliseconds=self.duration)
        timezone.make_aware(self.start)
        timezone.make_aware(self.end)
        self.release()
        self._cap = None
        self.final_frame = None

    @property
    def cap(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.path)
        return self._cap
    
    @cached_property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    @cached_property
    def frame_count(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    @cached_property
    def duration(self):
        return self.frame_count / self.fps * 1000

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
        self._cap = None
    
    @cached_property
    def filename(self):
        return self.path.split('/')[-1].split('.')[0]
    
    @property
    def pos_milli(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)
    
    @property
    def pos_frame(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    @property
    def pos_datetime(self):
        return self.start + timedelta(milliseconds=self.pos_milli)
    
    def seek_frame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
    def seek_milli(self, milli):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, milli)

    def clip_by_frame(self, outfile, start_frame=0, end_frame=None):
        return self.clip_by_milli(
            outfile, 
            start_milli=start_frame / self.fps * 1000,
            end_milli=None if end_frame is None else end_frame / self.fps * 1000
        )

    def clip_by_milli(self, outfile, start_milli=0, end_milli=None):
        end_str = ""
        if end_milli is not None:
            end_str = f"-to '{end_milli}ms'"
        os.system(f"{FFMPEG_BASE}  -ss '{start_milli}ms' {end_str} -i {self.path} -enc_time_base:v 1:24 {outfile}")

    def __str__(self) -> str:
        return self.path


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


def concat_clip(clip1_path, clip2_path):
    """Concatenates a video clip onto another video clip."""
    os.system(f'echo "file {clip1_path}" >> merge.txt')
    os.system(f'echo "file {clip2_path}" >> merge.txt')
    os.system(f"{FFMPEG_BASE} -f concat -safe 0 -i merge.txt -c copy merge.mp4")
    os.system(f"mv merge.mp4 {clip1_path}")
    os.system(f"rm -f merge.txt {clip2_path}")