import logging
import os
from dataclasses import dataclass
from enum import Enum
from json import load

import cv2
import numpy as np
from PIL import Image

from detection.models import Clip, ClipFragment
from detection.utils import image_from_array



@dataclass
class DetectorParams:
    camera: str = ""
    buffer: int = 5
    minlength: int = 5
    upper: float = 5
    lower: float = 1
    step: int = 6


class Motion(Enum):
    STILL = 0
    UNKNOWN = 1
    MOTION = 2


class Detector:
    """Detector takes a batch of videos processing tasks and detects motion in them, producing clip stubs."""

    def __init__(self, detection, handlers, logger=None, **params) -> None:
        
        if logger is None:
            self.logger = logging.getLogger('django')
        else:
            self.logger = logger

        self.detection = detection
        self.params = DetectorParams(params)
        self.handlers = handlers.order_by("video__start")

        assert handlers.values("batch__camera").distinct().count() == 1
        self.camera = handlers.first().camera

        self.clip = None
        self.counter = 0
        self.data = {}

    def process_frame(self, frame, box):
        """Perform preprocessing operations on an image"""
        (x1, y1), (x2, y2) = box.coords
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
        detect0 = self.process_frame(frame0, self.detection.detect_area)
        detect1 = self.process_frame(frame1, self.detection.detect_area)
        detect_score = self.compare_images(detect0, detect1)
        if detect_score >= self.params.upper:
            return Motion.MOTION
        elif detect_score <= self.params.lower:
            return Motion.STILL
        else:
            return Motion.UNKNOWN

    def detect_loop(self, start=0):
        """
        Loops through each video processing task and runs the detect code on it. Also check for
        continuous detect actions that straddle clips, if clips are sequential.
        """
        self.logger.info(f"Processing {self.handlers.count()} videos...")
        previous_end = None
        for handler in self.handlers.all():
            if start > self.counter:
                continue
            gap = previous_end is None or handler.video.start > previous_end
            if gap and self.clip is not None:
                self.logger.warning(f"  Found gap, unable to finish clip: {self.clip.outfile}")
                self.clip = None
            self.process_video(handler)
            previous_end = handler.video.end  # This should be populated at the end of the video capture
            handler.release()
            self.counter += 1

    def process_video(self, handler):
        """Loop through a single video and look for clips."""
        self.logger.info(f"  Processing video {self.counter + 1} of {self.handlers.count()}: {handler.video.filename}")
        tail_frames = handler.video.fps * self.params.buffer
        minlength = self.params.buffer + self.params.minlength 
        stills = 0
        fragment = None
        if self.clip is not None:
            self.logger.info(f"    Found unfinished clip, seeking matching frame...")
            frame0, fragment = self.seek_matching_image(handler.video, self.clip.last_fragment.end_frame)
        else:
            frame0 = handler.read()
        
        handler.seek_frame(frame0.frame_number + self.params.step - 1)
        frame1 = handler.read()
            
        while frame1 is not None:
            motion = self.detect_motion(frame0.image, frame1.image)

            if fragment is None and motion == Motion.MOTION:  # Start capturing
                self.clip = Clip(detection=self.detection)
                fragment = ClipFragment(video=handler.video, clip=self.clip, start=frame1.milliseconds)
                fragment.buff_start(self.params.buffer, save=False)
            elif fragment is not None:
                if motion == Motion.MOTION:
                    stills = 0
                elif motion == Motion.STILL:
                    stills += 1
                    if stills >= tail_frames:  # Stop Capturing
                        stills = 0
                        fragment.end = frame1.milliseconds
                        if fragment.duration >= minlength * 1000:
                            self.clip.save()
                            fragment.set_end_frame(frame1.image, save=False)
                            fragment.save()

                        fragment = None
                        self.clip = None
            
            frame0 = frame1
            frame1 = handler.read()

        # Cycle through the rest of the frames
        while frame0 is not None:
            end_frame = frame0
            frame0 = handler.read()

        # Stash an unfinished clip
        if fragment is not None:
            self.clip.save()
            fragment.set_end_frame(end_frame.image, save=False)
            fragment.save()

    def seek_matching_image(self, handler, image_match):
        """Find the first frame that matches the given image and set up a clip at that point."""
        
        # Get the video offset
        os.system(f'ffprobe -show_entries stream=codec_type,start_time -v 0 -of json {handler.file.path} >> offsets.json')
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

            frame = handler.read()
            if frame is None:
                raise Exception("Did not find matching frame.")

            diff = self.compare_images(image_match, frame.image)
            if diff == 0.0:
                frame = handler.read()
                fragment = ClipFragment(video=handler.video, start=frame.milliseconds, clip=self.clip, index=self.clip.fragments.count())
                fragment.start = fragment.start + offset
                break
        
        return frame, fragment
        

class ExclusionDetector(Detector):
    """
    ExclusionDetector is a detector class that allows for the exclusion of a region of the frame.
    
    If movement is detected in the detection region as well as in the exclusion region, it will be ignored.
    """    

    def detect_motion(self, image0, image1):
        detect_classification = super().detect_motion(image0, image1)

        exclude0 = self.process_frame(image0, self.detection.exclude_area)
        exclude1 = self.process_frame(image1, self.detection.exclude_area)
        exclude_score = self.compare_images(exclude0, exclude1)
        if exclude_score > self.params.upper:
            return Motion.STILL
        elif exclude_score < self.params.lower:
            return detect_classification
        else:
            return Motion.UNKNOWN
