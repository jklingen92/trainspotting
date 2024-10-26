import os
import cv2
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from functools import cached_property
from matplotlib.patches import Rectangle
from django.utils import timezone

FFMPEG_BASE = "ffmpeg -hide_banner -loglevel repeat+info"

class ImageInterface:
    def __init__(self, image):
        self.image = image
        self.bounding_box = None

    class TwoClickSelector:
        def __init__(self, ax, callback):
            self.ax = ax
            self.callback = callback
            self.clicks = []
            self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            if event.inaxes != self.ax:
                return
            self.clicks.append((int(event.xdata), int(event.ydata)))
            if len(self.clicks) == 1:
                print("First corner selected. Click to select the second corner.")
            if len(self.clicks) == 2:
                self.callback(self.clicks)
                self.ax.figure.canvas.mpl_disconnect(self.cid)
                plt.close()

    def draw_rectangle(self, clicks):
        self.bounding_box = clicks

    def get_bounding_box(self):
        while True:
            # Create the main figure and axis for selection
            fig, ax = plt.subplots()
            ax.imshow(self.image)
            ax.set_title("Click to select two corners of the rectangle")

            # Create the TwoClickSelector
            selector = self.TwoClickSelector(ax, self.draw_rectangle)
            plt.show()

            # Create a new figure to display the result
            fig, ax = plt.subplots()
            ax.imshow(self.image)

            # Draw the rectangle
            (x1, y1), (x2, y2) = self.bounding_box
            width = x2 - x1
            height = y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            plt.title('Image with Selected Rectangle')
            plt.show(block=False)

            # Ask for confirmation
            confirmation = input("Does this bounding box look correct? Y/n").lower().strip() or "y"
            plt.close()

            if confirmation == 'y':
                break

        return self.bounding_box

    def display_bounding_box(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image)

        (x1, y1), (x2, y2) = self.bounding_box
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        plt.title('Final Image with Selected Rectangle')
        plt.show()


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
    
    def seek_frame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
    def seek_milli(self, milli):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, milli)

    def clip(self, outfile, start_frame=0, end_frame=None):
        start = start_frame / self.fps * 1000
        end_str = ""
        if end_frame is not None:
            end_str = f"-to '{end_frame / self.fps * 1000}ms'"
        os.system(f"{FFMPEG_BASE}  -ss '{start}ms' {end_str} -i {self.path} {outfile}")


    def __str__(self) -> str:
        return self.path


class Clip:
    def __init__(self, start) -> None:
        self.start = start
        self.end = None
        self.stills = 0

        self.merge_path = None

    def buff_start(self, buffer):
        self.start = max(self.start - (buffer * 1000), 0)

    @property
    def duration(self):
        return self.end - self.start

    def dest(self, video):
        clip_datetime = video.start + timedelta(milliseconds=self.start)
        return f"{clip_datetime.strftime('%F_%T')}.mp4"


def milli2timestamp(milliseconds):
    return str(timedelta(milliseconds=milliseconds))
