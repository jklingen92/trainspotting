from datetime import timedelta
from functools import cached_property
import os
import cv2
from django.db import models
from django.conf import settings

from trainspotting.utils import concat_clip


class Video(models.Model):
    """Video stores metadata on a video as well as a link to the video file. It contains several utility functions for managing the file."""

    file = models.FileField(null=True)
    
    filename = models.CharField(max_length=240)
    camera = models.CharField(max_length=120)
    start = models.DateTimeField()
    fps = models.PositiveSmallIntegerField()
    frame_count = models.PositiveSmallIntegerField()

    @staticmethod
    def null_file_decorator(f):
        def inner_function(self, *args, **kwargs):
            return None if self.file == None else f(self, *args, **kwargs)
        return inner_function

    @null_file_decorator
    def init(self, save=True):
        """Populates values on instantiation."""
        self.filename = self.file.path.split('/')[-1].split('.')[0]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if save:
            self.save()

    @cached_property
    @null_file_decorator
    def cap(self):
        return cv2.VideoCapture(self.file.path)
    
    @property
    def duration(self):
        """Duration in milliseconds"""
        return self.frame_count / self.fps * 1000

    @property
    def end(self):
        return self.start + timedelta(milliseconds=self.duration)

    @null_file_decorator
    def read(self):
        return self.cap.read()
    
    @null_file_decorator
    def release(self):
        self.cap.release()

    @property
    @null_file_decorator
    def pos_milli(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)
    
    @property
    @null_file_decorator
    def pos_frame(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    @property
    @null_file_decorator
    def pos_datetime(self):
        return self.start + timedelta(milliseconds=self.pos_milli)
    
    @null_file_decorator
    def seek_frame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
    @null_file_decorator
    def seek_milli(self, milli):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, milli)

    @null_file_decorator
    def clip_by_milli(self, outfile, start_milli=0, end_milli=None):
        end_str = ""
        if end_milli is not None:
            end_str = f"-to '{end_milli}ms'"
        os.system(f"{settings.FFMPEG_BASE}  -ss '{start_milli}ms' {end_str} -i {self.file.path} -enc_time_base:v 1:24 {outfile}")
        return outfile

    def __str__(self) -> str:
        return self.filename


class Clip(models.Model):

    file = models.FileField(null=True)
    

class ClipStub(models.Model):

    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="clip_stubs")
    start = models.FloatField()  # milli
    
    end = models.FloatField(null=True)  # milli
    clip = models.ForeignKey(Clip, null=True, on_delete=models.SET_NULL, related_name="stubs")
    merge_to = models.ForeignKey("detection.ClipStub", null=True, on_delete=models.SET_NULL)
    end_frame = models.ImageField(null=True)

    def buff_start(self, buffer, save=True):
        self.start = max(self.start - (buffer * 1000), 0)
        if save:
            self.save()

    @property
    def duration(self):
        return self.end - self.start
    
    @property
    def start_datetime(self):
        return self.video.start + timedelta(milliseconds=self.start)

    @property
    def outfile(self):
        return f"{self.start_datetime.strftime('%F_%H%M%S')}.mp4"

    def clip(self, destination):
        """Extract a clip from a video, save it, and link it."""
        destination = os.path.join(destination, self.outfile)
        f = self.video.clip_by_milli(destination, start_milli=self.start, end_milli=self.end)
        if f is None:
            raise Exception("Video has no file attached")
        else:
            self.clip = Clip.objects.create(file=f)
            self.save()

    def clip_and_merge(self, clip):
        """Extract a clip from a video and merge it to an existing clip."""
        f = self.video.clip_by_milli(self.outfile, start_milli=self.start, end_milli=self.end)
        if f is None:
            raise Exception("Video has no file attached")
        else:
            concat_clip(clip.file.path, f.path)
            self.clip = clip
            self.save()

    def __str__(self) -> str:
        return f"{self.video.filename} [{timedelta(milliseconds=self.start)} - {timedelta(milliseconds=self.end) if self.end else 'End'}]{' (to be merged)' if self.merge_to else ''}"
    
