

import time
import cv2


class CapturePipeline:
    def __init__(self, exposure=450000, sensor_mode=0, warmup_frames=15):
        self.exposure = exposure
        self.warmup_frames = warmup_frames

        if sensor_mode in [0, 1]:
            self.width, self.height = 3840, 2160  # 4K
        elif sensor_mode == 2:
            self.width, self.height = 1920, 1080  # 1080p
        else:
            raise Exception(f"Invalid sensor mode: {sensor_mode}")

        pipeline = (
            f'nvarguscamerasrc sensor-mode={sensor_mode} exposuretimerange="{exposure} {exposure}" ! '
            f'video/x-raw(memory:NVMM), format=NV12, width={self.width}, height={self.height}, '
            f'framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! '
            f'video/x-raw, format=BGR ! appsink'
        )

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            raise Exception("Failed to open camera with GStreamer pipeline")
        
        for i in range(warmup_frames):
            ret, _ = self.cap.read()
            if not ret:
                raise Exception(f"Failed to grab frame during warmup ({i}/{warmup_frames})")
                
            # Sleep a bit between frames to give more time for adjustment
            time.sleep(0.1)

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
