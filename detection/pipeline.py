

import time
import cv2


class GStreamerPipeline:
    def __init__(self, sensor_mode=0, framerate=30, capture_class=cv2.VideoCapture,):
        self.sensor_mode = sensor_mode
        self.framerate = framerate
        self.capture_class = capture_class
        if sensor_mode in [0, 1]:
            self.width, self.height = 3840, 2160  # 4K
        elif sensor_mode == 2:
            self.width, self.height = 1920, 1080  # 1080p
        else:
            raise Exception(f"Invalid sensor mode: {sensor_mode}")
      
    def open_capture(self, exposure=450000, warmup_frames=30, **kwargs):
        cap = self.capture_class(self.get_capture_pipeline_str(exposure), cv2.CAP_GSTREAMER, **kwargs)
        if not cap.isOpened():
            cap.release()
            raise Exception("Failed to open camera with GStreamer pipeline")
        
        for i in range(warmup_frames):
            ret, _ = cap.read(ignore_motion=True)
            if not ret:
                raise Exception(f"Failed to grab frame during warmup ({i}/{warmup_frames})")
                
            # Sleep a bit between frames to give more time for adjustment
            time.sleep(0.1)

        return cap
    
    def open_output(self, output_path, bitrate=50000):

        out = cv2.VideoWriter(self.get_output_pipeline_str(output_path, bitrate), cv2.CAP_GSTREAMER, 0, float(self.framerate), (self.width, self.height))
        if not out.isOpened():
            out.release()
            raise Exception(f"Failed to open video writer with GStreamer pipeline for {output_path}")
        return out

    def get_capture_pipeline_str(self, exposure):
        return (
            f'nvarguscamerasrc sensor-mode={self.sensor_mode} exposuretimerange="{exposure} {exposure}" !'
            f'video/x-raw(memory:NVMM), format=NV12, width={self.width}, height={self.height}, '
            f'framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! appsink max-buffers=2 drop=true sync=false'
        )
    
    def get_output_pipeline_str(self, output_path, bitrate):
        return (
            f'appsrc ! videoconvert ! nvv4l2h264enc profile=4 bitrate={bitrate} speed-preset=ultra-fast !'
            f'video/x-h264, width={self.width}, height={self.height}, framerate={self.framerate}/1 ! '
            f'queue ! h264parse ! mp4mux ! filesink location={output_path}'
        )
