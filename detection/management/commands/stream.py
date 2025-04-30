# management/commands/camerastream.py

import os
import socket
from detection.camera_stream import OpenCVGstreamerStream
import numpy as np  # Required for blank frame
import sys
import importlib.util
from django.conf import settings
from django.urls import path
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.http import StreamingHttpResponse
from django.template.response import TemplateResponse
from django.urls import path
from django.views.decorators.gzip import gzip_page


# Global camera stream instance
camera_stream = None

def stream_view(request):
    """View function for the camera stream page"""
    return TemplateResponse(request, 'stream.html', {})

@gzip_page
def video_feed(request):
    """View function for the video feed"""
    global camera_stream
    return StreamingHttpResponse(
        camera_stream.generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

class Command(BaseCommand):
    help = 'Starts an OpenCV+GStreamer camera stream server for remote camera focus adjustment'
    
    def add_arguments(self, parser):
        parser.add_argument('--camera-id', type=int, default=0,
                            help='Camera device ID for USB camera (default: 0)')
        parser.add_argument('--width', type=int, default=1920,
                            help='Stream width in pixels (default: 1920)')
        parser.add_argument('--height', type=int, default=1080,
                            help='Stream height in pixels (default: 1080)')
        parser.add_argument('--fps', type=int, default=30,
                            help='Target frames per second (default: 30)')
        parser.add_argument('--port', type=int, default=8000,
                            help='HTTP server port (default: 8000)')
        parser.add_argument('--csi', action='store_true',
                            help='Use CSI camera instead of USB camera')
        parser.add_argument('--sensor-id', type=int, default=0,
                            help='Sensor ID for CSI camera (default: 0)')
        parser.add_argument('--flip-method', type=int, default=0,
                            help='Camera flip method: 0=none, 1=counterclockwise, 2=180, 3=clockwise (default: 0)')
    
    def handle(self, *args, **options):
        
        # Initialize the global camera stream
        global camera_stream
        camera_stream = OpenCVGstreamerStream(
            camera_id=options['camera_id'],
            width=options['width'],
            height=options['height'],
            fps=options['fps'],
            is_csi=options['csi'],
            sensor_id=options['sensor_id'],
            flip_method=options['flip_method']
        )
        
        # Start the OpenCV-GStreamer pipeline
        if not camera_stream.start_capture():
            self.stdout.write(self.style.ERROR('Failed to start camera capture'))
            return
        
        # Create a temporary template for the stream view
        os.makedirs(os.path.join(settings.BASE_DIR, 'templates'), exist_ok=True)
        with open(os.path.join(settings.BASE_DIR, 'templates', 'stream.html'), 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Jetson Camera Stream</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        margin: 0; 
                        padding: 20px; 
                        text-align: center; 
                        background-color: #f0f0f0; 
                    }
                    h1 { color: #333; }
                    .container { 
                        max-width: 100%; 
                        margin: 0 auto; 
                    }
                    .stream-container {
                        position: relative;
                        margin-top: 20px;
                    }
                    img { 
                        max-width: 100%; 
                        border: 1px solid #ddd; 
                        background-color: #ddd;
                    }
                    .controls {
                        margin-top: 20px;
                        padding: 10px;
                        background-color: #fff;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Jetson Camera Stream</h1>
                    <div class="stream-container">
                        <img src="/video_feed" alt="Camera Stream">
                    </div>
                    <div class="controls">
                        <p>Use this stream to adjust your camera's focus</p>
                    </div>
                </div>
            </body>
            </html>
            """)
        
        # Configure URLs
        
        
        # Store the original URLconf
        original_urlconf = settings.ROOT_URLCONF
        
        # Create a new module for the temporary URLconf
        
        spec = importlib.util.spec_from_loader('temp_urls', loader=None)
        temp_urls = importlib.util.module_from_spec(spec)
        sys.modules['temp_urls'] = temp_urls
        
        # Define the temporary urlpatterns
        temp_urls.urlpatterns = [
            path('', stream_view, name='stream'),
            path('video_feed', video_feed, name='video_feed'),
        ]
        
        # Use the temporary URLconf
        settings.ROOT_URLCONF = 'temp_urls'
        
        # Get the local IP address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            self.stdout.write(self.style.SUCCESS(f"Camera stream available at: http://{local_ip}:{options['port']}"))
        except:
            self.stdout.write(self.style.SUCCESS(f"Camera stream available at: http://YOUR_JETSON_IP:{options['port']}"))
        
        # Run the development server
        try:
            call_command('runserver', f"0.0.0.0:{options['port']}")
        except KeyboardInterrupt:
            # Restore the original URLconf on exit
            settings.ROOT_URLCONF = original_urlconf
            
            # Stop the camera stream
            if camera_stream is not None:
                camera_stream.stop_capture()
            
            self.stdout.write(self.style.SUCCESS('Camera stream stopped'))