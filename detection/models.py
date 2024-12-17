from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
import os
import cv2
from django.db import models
from django.conf import settings
from numpy import ndarray
from django.utils import timezone
from django.utils.text import slugify
from django_extensions.db.models import TimeStampedModel
from PIL import Image

from detection.utils import ImageInterface
from trainspotting.utils import concat_clips, display_tiles


@dataclass
class Frame:
    """Frame stores data from a video capture frame."""

    frame_number: int
    milliseconds: float
    image: ndarray


class Camera(TimeStampedModel):
    """Camera tracks info on specific camera locations."""

    name = models.CharField(max_length=120, unique=True)
    address = models.CharField(max_length=200)

    @property
    def video_destination(self):
        return os.path.join(settings.MEDIA_ROOT, "videos", slugify(self.name))


class VideoBatch(TimeStampedModel):
    """Videobatch logs info on a single batch import."""

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="batches")


    def __str__(self):
        videos = self.videos.order_by("start")
        first = videos.first().start.strftime('%F_%H%M%S')
        last = videos.last().start.strftime('%F_%H%M%S')
        return f"Batch ({first} - {last})"


class Video(TimeStampedModel):
    """Video stores metadata on a video that will persist after the video file is deleted."""

    filename = models.CharField(max_length=240, unique=True)
    start = models.DateTimeField()
    fps = models.PositiveSmallIntegerField()
    batch = models.ForeignKey(VideoBatch, on_delete=models.CASCADE, related_name="videos")

    # Populated after stepping through the whole video once
    frame_count = models.PositiveIntegerField(null=True)
    duration = models.FloatField(null=True)

    @property
    def camera(self):
        return self.batch.camera

    @property
    def end(self):
        return self.start + timedelta(milliseconds=self.duration)

    def milli_2_datetime(self, milliseconds):
        """Return a datetime object representing a millisecond offset in the video based on the start datetime"""
        return self.start + timedelta(milliseconds=milliseconds)

    def __str__(self) -> str:
        return self.filename


class VideoHandler(TimeStampedModel):
    """
    VideoHandler is a handler for an imported video file. 
    It is designed to be deleted along with the full video file.
    """

    file = models.FileField(unique=True)
    batch = models.ForeignKey(VideoBatch, on_delete=models.CASCADE, related_name="handlers")
    video = models.OneToOneField(Video, null=True, on_delete=models.SET_NULL, related_name="handler")
    processed = models.BooleanField(default=False)

    @property
    def camera(self):
        return self.batch.camera

    def init(self, start=None):
        """Creates a Video instance and populates it with simple values."""
        filename = self.file.path.split('/')[-1].split('.')[0]
        if start is None:  # Parse the video filename format "VID_20240101_163025" 
            start = timezone.make_aware(datetime.strptime(filename, "VID_%Y%m%d_%H%M%S"))
        
        self.video = Video.objects.create(
            batch=self.batch,
            filename=filename,
            start=start,
            fps = self.cap.get(cv2.CAP_PROP_FPS),
        )
        self.save(update_fields=["video"])

    @cached_property
    def cap(self):
        return cv2.VideoCapture(self.file.path)
    
    def read(self):
        """
        Reads the next frame of the video and passes it off in the Frame dataclass.

        The reason we do this is because certain readable properties of the VideoCapture 
        object are only reliable immiediately after the read operation, such as frame number
        and millisecond position.

        Additionally, we stash the total frame count and duration when we read the last frame, 
        because readable props like frame count are also not reliable.
        """
        success, img = self.cap.read()
        if not success:
            if not self.processed:  # Populate the frame count and duration
                self.video.frame_count = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.seek_frame(self.video.frame_count - 1)
                self.cap.read()
                self.video.duration = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                self.video.save(update_fields=["frame_count", "duration"])
                self.processed = True
                self.save(update_fields=["processed"])
                self.cap.read()
            return None
        return Frame(
            int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
            self.cap.get(cv2.CAP_PROP_POS_MSEC),
            img
        )
    
    def release(self):
        self.cap.release()
    
    def seek_frame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
    def seek_milli(self, milli):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, milli)

    def extract_by_milli(self, outfile, start_milli=0, end_milli=None):
        end_str = ""
        if end_milli is not None:
            end_str = f"-to '{end_milli}ms'"
        os.system(f"{settings.FFMPEG_BASE}  -ss '{start_milli}ms' {end_str} -i {self.file.path} -enc_time_base:v 1:24 {outfile}")
        return outfile


class BoundingBox(TimeStampedModel):

    left = models.PositiveIntegerField(default=0)
    top = models.PositiveIntegerField(default=0)
    right = models.PositiveIntegerField(null=True)
    bottom = models.PositiveIntegerField(null=True)

    @property
    def coords(self):
        return (self.left, self.top), (self.right, self.bottom)


class Detection(TimeStampedModel):
    """Detection logs info on a single batch detect."""

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="detections")
    sample = models.ImageField(upload_to="samples")
    view = models.CharField(max_length=120)
    _detect_area = models.ForeignKey(BoundingBox, null=True, on_delete=models.SET_NULL, related_name="detections")
    exclude_area = models.ForeignKey(BoundingBox, null=True, on_delete=models.SET_NULL, related_name="exclusions")
        
    @property
    def clip_destination(self):
        return os.path.join(settings.MEDIA_ROOT, "clips", slugify(self.camera.name), slugify(self.view))

    @property
    def detect_area(self):
        return self._detect_area or BoundingBox.objects.create(
            left=0, 
            top=0, 
            right=self.sample.width, 
            bottom=self.sample.height
        )
    
    @detect_area.setter
    def detect_area(self, value):
        self._detect_area = value

    def get_bounding_box(self, title, save=True):
        """Get a bounding box on the sample image"""
        sample = Image.open(self.sample.path)
        interface = ImageInterface(sample)
        bounding_box = interface.get_bounding_box(title=title)
        if bounding_box is None:
            return None
        else:
            (x1, y1), (x2, y2) = bounding_box
            bounding_box = BoundingBox(left=x1, top=y1, right=x2, bottom=y2)
            if save:
                bounding_box.save()
            return bounding_box

    def set_detect_area(self, save=True):
        """Set a bounding box to detect motion in."""
        detect_area = self.get_bounding_box(save=save)
        if detect_area:
            self.detect_area = detect_area
        else:
            self.detect_area = BoundingBox.objects.get_or_create(
                left=0, 
                top=0, 
                right=self.sample.size[0], 
                bottom=self.sample.size[1]
            )
        self.save(update_fields=['_detect_area'])

    def set_exclude_area(self, save=True):
        """Set a bounding box to exclude motion in."""
        self.exclude_area = self.get_bounding_box(save=save)
        self.save(update_fields=['exclude_area'])


class Clip(TimeStampedModel):
    """A single clip of motion. This may span more than one video."""
    
    detection = models.ForeignKey(Detection, on_delete=models.CASCADE, related_name="clips")
    file = models.FileField(null=True, upload_to="clips")

    @cached_property
    def duration(self):
        return sum(fragment.duration for fragment in self.fragments.all())
    
    @cached_property
    def start_datetime(self):
        return self.first_fragment.start_datetime

    @property
    def outfile(self):
        return f"{self.start_datetime.strftime('%F_%H%M%S')}.mp4"
    
    def display_frames(self):
        """Displays 5 frames from the clip, the first, last, 5s, 5s from the end, and center."""
        cap = cv2.VideoCapture(self.file.path)
        _, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
        _, frame2 = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, self.duration / 2)
        _, frame3 = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
        _, frame4 = cap.read()
        frame5 = self.last_fragment.end_frame
        display_tiles([frame1, frame2, frame3, frame4, frame5], ["First", "5s", "Middle", "-5s", "Last"])


    def extract(self):
        """Extract fragments from videos and merge them if necessary."""
        if not os.path.exists(self.detection.clip_destination):
            os.makedirs(self.detection.clip_destination)
        fragments_to_merge = []
        for fragment in self.fragments.order_by("index"):
            dest = os.path.join(self.detection.clip_destination, self.outfile)
            if fragment.index > 0:
                dest = dest + f"_{fragment.index}"
            fragments_to_merge.append(dest)
            fragment.extract(dest)
        if len(fragments_to_merge) > 1:
            concat_clips(fragments_to_merge)
        
        self.file = fragments_to_merge[0]
        self.save(update_fields=["file"])

    @property
    def first_fragment(self):
        return self.fragments.get(index=0)

    @property
    def last_fragment(self):
        return self.fragments.order_by("index").last()

    def __str__(self) -> str:
        return f"{self.outfile} ({'not ' if not self.file else ''}extracted)"
    

class ClipFragment(TimeStampedModel):
    """A single fragment of a clip from a video. More than one fragment can exist in a single video."""

    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="clips")
    clip = models.ForeignKey(Clip, on_delete=models.CASCADE, related_name="fragments")
    index = models.PositiveSmallIntegerField(default=0)
    start = models.FloatField()  # milli
    end = models.FloatField(null=True)  # milli
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

    def extract(self, destination):
        self.video.handler.extract_by_milli(destination, start_milli=self.start,  end_milli=self.end)