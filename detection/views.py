import base64
import cv2
from django.views import View
import time
import logging
from django.http import HttpResponse

from detection.capture_pipeline import CapturePipeline

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SnapshotView(View):
    """View that captures and displays a snapshot from the camera"""
    
    def get(self, request):
        pipeline = CapturePipeline(sensor_mode=0, exposure=450000, warmup_frames=15)    
        # Capture the image
        ret, frame = pipeline.cap.read()
        
        # Convert to JPEG and then base64 for embedding in HTML
        _, buffer = cv2.imencode('.png', frame)
        png_base64 = base64.b64encode(buffer).decode('utf-8')

        # Create HTML with the embedded image
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Jetson Camera Snapshot</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; margin: 20px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                button {{ margin: 10px; padding: 10px 20px; background-color: #4CAF50; 
                        color: white; border: none; border-radius: 4px; cursor: pointer; }}
                button:hover {{ background-color: #45a049; }}
                .timestamp {{ margin-top: 10px; color: #666; }}
            </style>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body>
            <div class="container">
                <h1>Jetson Camera Snapshot</h1>
                <div class="timestamp">Captured at: {time.strftime("%Y-%m-%d %H:%M:%S")}</div>
                <div>
                    <img src="data:image/png;base64,{png_base64}" alt="Camera Snapshot">
                </div>
                <div>
                    <button onclick="window.location.reload();">Take New Snapshot</button>
                    <button onclick="downloadImage()">Download Image</button>
                </div>
            </div>
            
            <script>
                function downloadImage() {{
                    // Create a link element
                    const link = document.createElement('a');
                    
                    // Set link properties
                    link.href = "data:image/png;base64,{png_base64}";
                    link.download = "jetson_snapshot_{time.strftime("%Y%m%d_%H%M%S")}.png";
                    
                    // Append to the body, click it, and remove it
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }}
            </script>
        </body>
        </html>
        """
        
        return HttpResponse(html)
    
import cv2
import numpy as np
import threading
import time
from django.views import View
from django.http import StreamingHttpResponse, HttpResponse
from django.views.decorators import gzip
from django.utils.decorators import method_decorator

class CameraStream:
    """Class to handle the camera stream"""
    
    def __init__(self, sensor_mode=2, exposure=33333333):
        self.sensor_mode = sensor_mode
        self.exposure = exposure
        self.frame = None
        self.grabbed = False
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.video = None
        
    def start(self):
        """Start capturing from the camera"""
        if self.running:
            print("Camera is already running")
            return
            
        # Create the GStreamer pipeline
        self.pipeline = CapturePipeline(
            sensor_mode=self.sensor_mode,
            exposure=self.exposure,
            warmup_frames=15
        )
        self.video = self.pipeline.cap
    
   
        # Read a test frame
        self.grabbed, self.frame = self.video.read()
        if not self.grabbed or self.frame is None:
            self.video.release()
            raise RuntimeError("Could not grab initial frame from camera")
            
        print(f"Camera initialized successfully. Frame shape: {self.frame.shape}")
        
        # Start the background thread
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        return self
        
    def _update(self):
        """Update thread to continuously get frames from the camera"""
        while self.running:
            if not self.video.isOpened():
                print("Camera connection lost")
                self.running = False
                break
                
            try:
                grabbed, frame = self.video.read()
                
                if not grabbed or frame is None:
                    print("Failed to grab frame")
                    time.sleep(0.1)
                    continue
                    
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    
            except Exception as e:
                print(f"Error in camera thread: {e}")
                
            # Control frame rate
            time.sleep(0.033)  # ~30 FPS
            
    def get_frame(self):
        """Get the latest frame from the camera"""
        with self.lock:
            if not self.grabbed or self.frame is None:
                return None
            return self.frame.copy()
            
    def stop(self):
        """Stop the camera stream"""
        self.running = False
        
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        if self.video is not None and self.video.isOpened():
            self.video.release()
            
        print("Camera stream stopped")

# Global camera instance
camera_stream = None

def get_camera_stream():
    """Get or initialize the camera stream"""
    global camera_stream
    if camera_stream is None or not camera_stream.running:
        try:
            camera_stream = CameraStream().start()
        except Exception as e:
            print(f"Error initializing camera: {e}")
            raise
    return camera_stream

@method_decorator(gzip.gzip_page, name='dispatch')
class CameraStreamView(View):
    """Class-based view for camera streaming"""
    
    def get(self, request):
        """Handle GET request"""
        try:
            return StreamingHttpResponse(
                self._generate_frames(),
                content_type='multipart/x-mixed-replace; boundary=frame'
            )
        except Exception as e:
            print(f"Stream error: {e}")
            return HttpResponse(f"Stream error: {e}", status=500)
            
    def _generate_frames(self):
        """Generator for frames"""
        print("Starting frame generation")
        cam = get_camera_stream()
        
        while True:
            frame = cam.get_frame()
            
            if frame is None:
                # Generate an error frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No frame available", 
                           (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add timestamp to the frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Convert to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            if not ret:
                print("Failed to encode JPEG")
                time.sleep(0.1)
                continue
                
            # Yield the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                   
            # Control frame rate for the stream delivery
            time.sleep(0.033)  # ~30 FPS

class CameraPageView(View):
    """Class-based view for the camera page"""
    
    def get(self, request):
        """Handle GET request"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Jetson Camera Stream</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }
                img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
                .container { max-width: 800px; margin: 0 auto; }
                .controls { margin-top: 20px; }
                button { margin: 5px; padding: 8px 16px; background-color: #4CAF50; 
                        color: white; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #45a049; }
                .status { margin-top: 10px; padding: 10px; border-radius: 4px; 
                         background-color: #f8f9fa; display: inline-block; }
            </style>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script>
                window.onload = function() {
                    var img = document.getElementById('stream');
                    var status = document.getElementById('status');
                    
                    img.onload = function() {
                        status.textContent = 'Stream Active';
                        status.style.backgroundColor = '#d4edda';
                        status.style.color = '#155724';
                    };
                    
                    img.onerror = function() {
                        status.textContent = 'Stream Error - Reconnecting...';
                        status.style.backgroundColor = '#f8d7da';
                        status.style.color = '#721c24';
                        setTimeout(refreshStream, 2000);
                    };
                    
                    // Periodic check
                    setInterval(function() {
                        if (img.complete && img.naturalHeight !== 0) {
                            status.textContent = 'Stream Active';
                            status.style.backgroundColor = '#d4edda';
                            status.style.color = '#155724';
                        }
                    }, 5000);
                };
                
                function refreshStream() {
                    var img = document.getElementById('stream');
                    img.src = '/camera/stream/?' + new Date().getTime();
                    document.getElementById('status').textContent = 'Reconnecting...';
                }
                
                function takeSnapshot() {
                    var img = document.getElementById('stream');
                    var canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    
                    var ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    
                    var dataUrl = canvas.toDataURL('image/png');
                    
                    var link = document.createElement('a');
                    link.href = dataUrl;
                    link.download = 'jetson_snapshot_' + new Date().toISOString().replace(/:/g, '-') + '.png';
                    link.click();
                }
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Jetson Camera Stream</h1>
                <div id="status" class="status">Connecting to stream...</div>
                <div>
                    <img id="stream" src="/camera/stream/" alt="Camera Stream">
                </div>
                <div class="controls">
                    <button onclick="refreshStream()">Refresh Stream</button>
                    <button onclick="takeSnapshot()">Take Snapshot</button>
                </div>
            </div>
        </body>
        </html>
        """
        return HttpResponse(html)

# Clean up on Django shutdown
def cleanup_camera():
    """Clean up camera resources on shutdown"""
    global camera_stream
    if camera_stream is not None:
        print("Shutting down camera...")
        camera_stream.stop()
        camera_stream = None

# Register cleanup handler
import atexit
atexit.register(cleanup_camera)