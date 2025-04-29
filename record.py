import cv2
import time
import os
import argparse
import traceback
import subprocess
import signal
import threading

def camera_capture(
    output_file='camera_test.mp4',
    duration=0,
    exposure=450000,
    auto_gain=True,
    resolution=(3840, 2160),
    framerate=30,
    bitrate=8000,
    rotation=0,
    sensor_mode=0,
    hdr_enable=False,
    preview=False
):
    """
    Run a camera test with specified settings.
    
    Args:
        output_file: Path to save the recorded video (ignored if preview=True)
        duration: Recording/preview duration in seconds (0 = run until Ctrl+C for preview)
        exposure: Exposure time in nanoseconds
        auto_gain: Whether to use auto gain (True) or fixed gain=1.0 (False)
        resolution: Width and height tuple
        framerate: Frames per second
        bitrate: Encoding bitrate in Kbps (ignored if preview=True)
        rotation: Rotation method (0=none, 1=90° clockwise, 2=180°, 3=90° counterclockwise)
        sensor_mode: Sensor mode (0-3)
        hdr_enable: Enable HDR mode
        preview: Whether to open a preview window instead of recording
    """
    try:
        # Configure camera parameters
        camera_params = []
        
        # Add sensor mode parameter
        camera_params.append(f"sensor-mode={sensor_mode}")
        
        # Add HDR toggle
        if hdr_enable:
            camera_params.append("wbmode=9")  # Special white balance mode for HDR
            camera_params.append("tnr-mode=2")  # Temporal noise reduction for HDR
            camera_params.append("ee-mode=2")  # Edge enhancement for HDR
            camera_params.append("aeantibanding=2")  # Anti-banding for HDR
        
        # Set fixed exposure (minimum)
        camera_params.append(f"exposuretimerange=\"{exposure} {exposure}\"")
        camera_params.append("aelock=true")  # Lock auto exposure
        
        # Configure gain
        if not auto_gain:
            camera_params.append("gainrange=\"1.0 1.0\"")  # Fixed gain at 1.0
            
        camera_param_str = " ".join(camera_params)
        
        if preview:
            # Create GStreamer pipeline for preview
            pipeline_elements = [
                f"nvarguscamerasrc {camera_param_str}",
                f"video/x-raw(memory:NVMM), width={resolution[0]}, height={resolution[1]}, framerate={framerate}/1",
                "nvvidconv",
                f"videoflip method={rotation}",
                "video/x-raw, format=BGRx",
                "videoconvert",
                f"textoverlay text=\"Camera Preview: Mode={sensor_mode}, HDR={'On' if hdr_enable else 'Off'}, Exp={exposure}ns\" "
                "valignment=top halignment=left font-desc=\"Sans, 24\"",
                "autovideosink"
            ]
            
            # Create the complete pipeline string
            pipeline_str = " ! ".join(pipeline_elements)
            
            print(f"Preview pipeline: {pipeline_str}")
            print(f"Settings: sensor_mode={sensor_mode}, hdr={'enabled' if hdr_enable else 'disabled'}, "
                  f"exposure={exposure}ns, auto_gain={auto_gain}, {resolution[0]}x{resolution[1]} @ {framerate}fps")
            print("Press Ctrl+C to exit preview")
            
            # Start GStreamer process
            process = subprocess.Popen(['gst-launch-1.0'] + pipeline_str.split(), 
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True,
                                      preexec_fn=os.setsid)  # Use process group for clean termination
            
            # Set up timer if duration is specified
            if duration > 0:
                def stop_after_duration():
                    time.sleep(duration)
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    print(f"Preview stopped after {duration} seconds")
                
                timer_thread = threading.Thread(target=stop_after_duration)
                timer_thread.daemon = True
                timer_thread.start()
            
            try:
                # Wait for process to complete or be terminated
                stdout, stderr = process.communicate()
                if process.returncode != 0 and process.returncode != -15:  # -15 is SIGTERM
                    print(f"GStreamer error (code {process.returncode}):")
                    print(stderr)
                    return False
                    
                return True
                
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\nStopping preview...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait()
                print("Preview stopped")
                return True
        
        else:
            # Original recording mode
            # Create GStreamer pipeline for camera capture
            pipeline = (
                f"nvarguscamerasrc {camera_param_str} ! "
                f"video/x-raw(memory:NVMM), width={resolution[0]}, height={resolution[1]}, "
                f"framerate={framerate}/1 ! "
                f"nvvidconv ! "
                f"videoflip method={rotation} ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink max-buffers=1 drop=true"
            )
            
            print(f"Camera pipeline: {pipeline}")
            
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                print("Failed to open camera!")
                return False
            
            print(f"Camera opened successfully")
            
            # Define the output GStreamer pipeline
            gst_out = (
                "appsrc ! "
                "video/x-raw, format=BGR ! "
                "videoconvert ! "
                "video/x-raw, format=I420 ! "
                f"x264enc speed-preset=ultrafast tune=zerolatency bitrate={bitrate} ! "
                "h264parse ! "
                "mp4mux ! "
                f"filesink location={output_file}"
            )
            
            # Read first frame to get dimensions
            ret, test_frame = cap.read()
            if not ret:
                print("Failed to read initial frame")
                cap.release()
                return False
                
            width = test_frame.shape[1]
            height = test_frame.shape[0]
            
            out = cv2.VideoWriter(
                gst_out, 
                cv2.CAP_GSTREAMER, 
                0,
                float(framerate),
                (width, height)
            )
            
            if not out.isOpened():
                print("Failed to create video writer!")
                cap.release()
                return False
            
            print(f"Recording to {output_file} for {duration} seconds")
            print(f"Settings: exposure={exposure}ns, auto_gain={auto_gain}, {width}x{height} @ {framerate}fps")
            
            frame_count = 0
            brightness_samples = []
            
            # Allow camera to adjust for the first second
            settle_time = time.time()
            while (time.time() - settle_time) < 1.0:
                ret, _ = cap.read()  # Discard frames while settling
                
            # Start recording
            start_time = time.time()  # Reset start time after settling
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to grab frame")
                    break
                    
                # Write frame to video
                out.write(frame)
                
                frame_count += 1
                
                # Calculate and store brightness samples
                if frame_count % 10 == 0:
                    brightness = frame.mean()
                    brightness_samples.append(brightness)
                    elapsed = time.time() - start_time
                    print(f"Frame {frame_count}, Brightness: {brightness:.1f}, Time: {elapsed:.1f}s", end="\r")
            
            print("")  # New line after progress updates
            
            # Calculate statistics
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Calculate brightness statistics if we have samples
            if brightness_samples:
                avg_brightness = sum(brightness_samples) / len(brightness_samples)
                min_brightness = min(brightness_samples)
                max_brightness = max(brightness_samples)
                print(f"Brightness stats - Avg: {avg_brightness:.1f}, Min: {min_brightness:.1f}, Max: {max_brightness:.1f}")
            
            print(f"Recorded {frame_count} frames in {elapsed:.2f} seconds ({actual_fps:.2f} fps)")
            
            out.release()
            cap.release()
            print(f"Video saved to {os.path.abspath(output_file)}")
            
            return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False

def list_camera_controls():
    """List all available camera controls using v4l2-ctl"""
    try:
        result = subprocess.run(['v4l2-ctl', '--list-ctrls-menus'], 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        print("Available camera controls:")
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Error listing camera controls: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run camera test or preview with various settings")
    parser.add_argument("--output", type=str, default="camera_test.mp4", 
                        help="Output video file path (for recording mode)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Recording/preview duration in seconds (0 = run until Ctrl+C for preview)")
    parser.add_argument("--exposure", type=int, default=450000,
                        help="Exposure time in nanoseconds")
    parser.add_argument("--no-auto-gain", action="store_true",
                        help="Disable auto gain (use fixed gain=1.0)")
    parser.add_argument("--width", type=int, default=3480,
                        help="Video width")
    parser.add_argument("--height", type=int, default=2160,
                        help="Video height")
    parser.add_argument("--framerate", type=int, default=30,
                        help="Recording/preview framerate")
    parser.add_argument("--bitrate", type=int, default=50000,
                        help="Encoding bitrate in Kbps (for recording mode)")
    parser.add_argument("--rotation", type=int, default=0,
                        help="Rotation method (0=none, 1=90° clockwise, 2=180°, 3=90° counterclockwise)")
    parser.add_argument("--sensor-mode", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Sensor mode (0-3)")
    parser.add_argument("--hdr", action="store_true",
                        help="Enable HDR mode")
    parser.add_argument("--preview", action="store_true",
                        help="Open a preview window instead of recording")
    parser.add_argument("--list-controls", action="store_true",
                        help="List all available camera controls and exit")
    
    args = parser.parse_args()
    
    # If requested, list all camera controls and exit
    if args.list_controls:
        list_camera_controls()
        exit(0)
    
    # Run the camera test with command line arguments
    camera_capture(
        output_file=args.output,
        duration=args.duration,
        exposure=args.exposure,
        auto_gain=not args.no_auto_gain,
        resolution=(args.width, args.height),
        framerate=args.framerate,
        bitrate=args.bitrate,
        rotation=args.rotation,
        sensor_mode=args.sensor_mode,
        hdr_enable=args.hdr,
        preview=args.preview
    )
