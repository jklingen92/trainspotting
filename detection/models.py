from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
import os
import cv2
from django.db import models
from django.conf import settings
from numpy import ndarray
from django.utils import timezone
from django_extensions.db.models import TimeStampedModel
from PIL import Image

from detection.utils import ImageInterface
from trainspotting.utils import concat_clips


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


class ImportTask(TimeStampedModel):
    """ImportTask logs info on a single batch import."""

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="import_tasks")


class Video(TimeStampedModel):
    """Video stores metadata on a video that will persist after the video file is deleted."""

    filename = models.CharField(max_length=240, unique=True)
    start = models.DateTimeField()
    fps = models.PositiveSmallIntegerField()
    import_task = models.ForeignKey(ImportTask, on_delete=models.CASCADE, related_name="videos")

    # Populated after stepping through the whole video once
    frame_count = models.PositiveIntegerField(null=True)
    duration = models.FloatField(null=True)

    @property
    def camera(self):
        return self.import_task.camera

    @property
    def end(self):
        return self.start + timedelta(milliseconds=self.duration)

    def milli_2_datetime(self, milliseconds):
        """Return a datetime object representing a millisecond offset in the video based on the start datetime"""
        return self.start + timedelta(milliseconds=milliseconds)

    def __str__(self) -> str:
        return self.filename


class VideoProcessingTask(TimeStampedModel):
    """
    VideoProcessingTask is a handler for an imported video file. 
    It is designed to be deleted along with the full video file.
    """

    file = models.FileField(unique=True)
    import_task = models.ForeignKey(ImportTask, on_delete=models.CASCADE, related_name="processing_tasks")
    video = models.OneToOneField(Video, null=True, on_delete=models.SET_NULL)
    processed = models.BooleanField(default=False)

    @property
    def camera(self):
        return self.import_task.camera

    def init(self, start=None):
        """Creates a Video instance and populates it with simple values."""
        filename = self.file.path.split('/')[-1].split('.')[0]
        if start is None:  # Parse the video filename format "VID_20240101_163025" 
            start = timezone.make_aware(datetime.strptime(filename, "VID_%Y%m%d_%H%M%S"))
        
        self.video = Video.objects.create(
            import_task=self.import_task,
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


class DetectTask(TimeStampedModel):
    """DetectTask logs info on a single batch detect."""

    import_task = models.ForeignKey(ImportTask, on_delete=models.PROTECT)
    sample = models.ImageField(upload_to="samples")
    view = models.CharField(max_length=120)
    _detect = models.ForeignKey(BoundingBox, null=True, on_delete=models.SET_NULL, related_name="detect_tasks")
    exclude = models.ForeignKey(BoundingBox, null=True, on_delete=models.SET_NULL, related_name="exclude_tasks")

    @property
    def camera(self):
        return self.import_task.camera
    
    @property
    def detect(self):
        return self._detect or BoundingBox.objects.create(
            left=0, 
            top=0, 
            right=self.sample.width, 
            bottom=self.sample.height
        )
    
    @detect.setter
    def detect(self, value):
        self._detect = value

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

    def set_detect_box(self, save=True):
        """Set a bounding box to detect motion in."""
        detect = self.get_bounding_box(save=save)
        if detect:
            self.detect = detect
        else:
            self.detect = BoundingBox.objects.get_or_create(
                left=0, 
                top=0, 
                right=self.sample.size[0], 
                bottom=self.sample.size[1]
            )
        self.save(update_fields=['detect'])

    def set_exclude_box(self, save=True):
        """Set a bounding box to exclude motion in."""
        self.exclude = self.get_bounding_box(save=save)
        self.save(update_fields=['exclude'])



class Clip(TimeStampedModel):
    
    detect_task = models.ForeignKey(DetectTask, on_delete=models.CASCADE, related_name="clips")
    file = models.FileField(null=True, upload_to="clips", unique=True)

    @cached_property
    def duration(self):
        return sum(fragment.duration for fragment in self.fragments.all())
    
    @cached_property
    def start_datetime(self):
        return self.first_fragment.start_datetime

    @property
    def outfile(self):
        return f"{self.start_datetime.strftime('%F_%H%M%S')}.mp4"

    @property
    def clip_destination(self):
        return os.path.join(settings.MEDIA_ROOT, "clips", self.detect_task.camera.name, self.outfile)

    def extract(self):
        """Extract fragments from videos and merge them if necessary."""
        fragments_to_merge = []
        for fragment in self.fragments.order_by("index"):
            if fragment.index == 0:
                dest = self.clip_destination
            else:
                dest = self.clip_destination + f"_{fragment.index}"
            fragments_to_merge += dest
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
        return f"{self.outfile} ({'not' if not self.file else ''} extracted)"
    


class ClipFragment(TimeStampedModel):
    """A single fragment of a clip from a video"""

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
        self.video.videoprocessingtask.extract_by_milli(destination, start_milli=self.start,  end_milli=self.end)