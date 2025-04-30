import os
import signal
import subprocess
from django.core.management.base import BaseCommand
from detection.pipeline import build_pipeline

class Command(BaseCommand):
    help = 'Opens an OpenCV preview window using GStreamer and the build_pipeline function'

    def handle(self, *args, **kwargs):

        # Build the GStreamer pipeline using the build_pipeline function
        pipeline_str = build_pipeline(
            camera_id=0,
            resolution=[3840, 2160],
            framerate=30,
            exposure=450000,
            bitrate=50000,
            auto_gain=True,
            preview=True
        )

        self.stdout.write(f"Preview pipeline: {pipeline_str}")
        self.stdout.write(f"Settings: exposure=450000ns, 3840x2160 @ 30fps")
        self.stdout.write("Press Ctrl+C to exit preview")

        process = subprocess.Popen(['gst-launch-1.0'] + pipeline_str.split(), 
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True,
                                      preexec_fn=os.setsid)  # Use process group for clean termination

        try:
            # Wait for process to complete or be terminated
            stdout, stderr = process.communicate()
            if process.returncode != 0 and process.returncode != -15:  # -15 is SIGTERM
                self.stdout.write(f"GStreamer error (code {process.returncode}):")
                self.stdout.write(stderr)
                return False
                
            return True
        
        except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                self.stdout.write("\nStopping preview...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait()
                self.stdout.write("Preview stopped")
                return True
