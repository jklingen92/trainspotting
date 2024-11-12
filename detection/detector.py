import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from json import load

import cv2
import numpy as np
from django.utils import timezone
from PIL import Image

from detection.models import ClipStub, Video
from detection.utils import ImageInterface


@dataclass
class DetectorParams:
    camera: str = ""
    buffer: int = 5
    minlength: int = 5
    upper: float = 5
    lower: float = 1


class Motion(Enum):
    STILL = 0
    UNKNOWN = 1
    MOTION = 2


class Detector:
    """Detector takes a batch of videos and detects motion in them, producing clips."""

    def __init__(self, video_paths, logger=None, **params) -> None:

        def log(msg, **kwargs):
            if logger is not None:
                logger(msg, **kwargs)
            else:
                print(msg)

        self.log = log
        self.params = DetectorParams(params)

        self.num_videos = len(video_paths)
        self.video_paths = self.sort_video_paths(video_paths)
        first_video = self.create_video(self.video_paths.pop(0), save=False)
        success, first_frame = first_video.read()
        if not success:
            raise Exception(f"First video has no frames!")
        first_video.release()

        self.first_frame_interface = ImageInterface(first_frame)
        self.detect_box = self.first_frame_interface.get_bounding_box(
            title="Select a region for detection, or leave blank to use the full frame",
            defaults=((0, 0), (self.first_frame_interface.width, self.first_frame_interface.height))
        )

        self.log(f"Processing {self.num_videos} videos...")
        self.videos = self.video_generator(video_paths)
        
        self.unfinished_stub = None
        self.counter = 0
        self.data = {}

    def sort_video_paths(self, video_paths):
        """Sort video paths."""
        return sorted(video_paths)
    
    def create_video(self, video_path, save=True):
        video = Video(file=video_path, camera=self.params.camera)
        video.init(save=False)
        video.start = timezone.make_aware(datetime.strptime(video.filename, "VID_%Y%m%d_%H%M%S"))
        if save:
            video.save()
        return video

    def video_generator(self, video_paths):
        """Create a lazy generator that returns Videos based on paths"""
        last_video_end = None
        for i, video_path in enumerate(video_paths):
            self.counter = i
            video = self.create_video(video_path)
            gap = last_video_end is None or last_video_end > video.start
            yield gap, video
            last_video_end = video.end
            video.release()

    def process_frame(self, frame, box):
        """Perform preprocessing operations on an image"""
        (x1, y1), (x2, y2) = box
        frame = frame[y1:y2, x1:x2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        return frame
    
    def compare_images(self, img0, img1):
        """Returns the simple mean difference between two images"""
        diff = cv2.absdiff(img0, img1)
        score = np.mean(diff)
        return score
    
    def detect_motion(self, frame0, frame1):
        """Compare two frames and return motion classification."""
        detect0 = self.process_frame(frame0, self.detect_box)
        detect1 = self.process_frame(frame1, self.detect_box)
        detect_score = self.compare_images(detect0, detect1)
        if detect_score >= self.params.upper:
            return Motion.MOTION
        elif detect_score <= self.params.lower:
            return Motion.STILL
        else:
            return Motion.UNKNOWN

    def detect_loop(self, save=True):
        """
        A generator that process each video, returning the resulting data.
        It wiped unfinished clips in the case of a gap in time.
        """
        for gap, video in self.videos:
            if gap and self.unfinished_stub is not None:
                self.log(f"  Found gap, unable to finish clip: {self.unfinished_stub.path}")
                self.unfinished = None
            stubs_to_create = self.process_video(video)
            video.release()
            if save:
                return ClipStub.objects.bulk_create(stubs_to_create)
            else:
                return stubs_to_create

    def process_video(self, video):
        """Loop through a single video and look for clips."""
        self.log(f"  Processing video {self.counter + 1} of {self.num_videos}: {video.file.path}", ending="\n")
        tail_frames = video.fps * self.params.buffer
        minlength = self.params.buffer + self.params.minlength 
        stills = 0
        stubs = []
        clip_stub = None
        if self.unfinished_stub is not None:
            self.log(f"    Found unfinished clip, seeking matching frame...")
            frame0, clip_stub = self.seek_matching_frame(video, self.unfinished_stub.end_frame)
            clip_stub.merge_to = self.unfinished_stub
            self.unfinished_stub = None
        else:
            success, frame0 = video.read()
        
        success, frame1 = video.read()
            
        while success:
            motion = self.detect_motion(frame0, frame1)

            if clip_stub is not None:
                status = f"    Capturing: {timedelta(milliseconds=video.pos_milli - clip_stub.start)} | Stills: {stills:4d}"
            else:
                status = "-" * 30
            self.log(f"    Progress: {timedelta(milliseconds=video.pos_milli)} | {status}", ending="\r")

            if clip_stub is None and motion == Motion.MOTION:  # Start capturing
                clip_stub = ClipStub(video=video, start=video.pos_milli)
                clip_stub.buff_start(self.params.buffer, save=False)
            elif clip_stub is not None:
                if motion == Motion.MOTION:
                    stills = 0
                elif motion == Motion.STILL:
                    stills += 1
                    if stills >= tail_frames:
                        stills = 0
                        clip_stub.end = video.pos_milli
                        if clip_stub.duration >= minlength * 1000:
                            stubs.append(clip_stub)
                        clip_stub = None
            
            frame0 = frame1
            success, frame1 = video.read()

        # Stash an unfinished clip
        if clip_stub is not None:
            clip_stub.end_frame = Image.from_array(frame0)
            self.unfinished_stub = clip_stub
            stubs.append(clip_stub)
        return stubs

    def seek_matching_frame(self, video, frame):
        """Find the first frame that matches the given frame and set up a clip at that point."""
        
        # Get the video offset
        os.system(f'ffprobe -show_entries stream=codec_type,start_time -v 0 -of json {video.path} >> offsets.json')
        with open("offsets.json") as f:
            offsets = load(f)
        os.system(f"rm -f offsets.json")
        
        video_offset = 0
        smallest_offset = np.inf
        for stream in offsets["streams"]:
            if stream["codec_type"] == "video":
                video_offset = float(stream["start_time"])
            smallest_offset = min(float(stream["start_time"]), smallest_offset)
        
        offset = (video_offset - smallest_offset) * 1000

        # Seek the frame
        while True:

            success, img = video.read()
            if not success:
                raise Exception("Did not find matching frame.")

            diff = self.compare_images(frame, img)
            if diff == 0.0:
                _, frame = video.read()
                clip_stub = ClipStub(video=video, start=video.pos_milli)
                clip_stub.start = clip_stub.start + offset
                break
        
        return frame, clip_stub
        

class ExclusionDetector(Detector):
    """
    ExclusionDetector is a detector class that allows for the exclusion of a region of the frame.
    
    If movement is detected in the detection region as well as in the exclusion region, it will be ignored.
    """    
    
    def __init__(self, video_paths, log=None, **params) -> None:
        super().__init__(video_paths, log, **params)
        self.exclude_box = self.first_frame_interface.get_bounding_box(
            title="Select a region for exclusion, or leave blank for None",
            color="red"
        )

    def detect_motion(self, frame0, frame1):
        detect_classification = super().detect_motion(frame0, frame1)

        exclude0 = self.process_frame(frame0, self.exclude_box)
        exclude1 = self.process_frame(frame1, self.exclude_box)
        exclude_score = self.compare_images(exclude0, exclude1)
        if exclude_score > self.params.upper:
            return Motion.STILL
        elif exclude_score < self.params.lower:
            return detect_classification
        else:
            return Motion.UNKNOWN
