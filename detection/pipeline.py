

def build_pipeline(camera_id, resolution, framerate, exposure, bitrate, auto_gain, preview=False):
    """
    Constructs a GStreamer pipeline string for capturing video from a camera.
    Args:
        camera_id (int): The ID of the camera to use.
        resolution (tuple): The resolution of the video in pixels (width, height).
        framerate (int): The frame rate of the video.
        exposure (int): The exposure time in nanoseconds.
        bitrate (int): The bitrate of the video in kbps.
        auto_gain (bool): Whether to use auto gain or not.
        preview (bool): Whether to include a preview window or not.
    Returns:
        str: The GStreamer pipeline string.
    """
    # Configure camera parameters
    camera_params = []
    
    # Add sensor mode parameter
    camera_params.append(f"sensor-mode=1")
    
    # Set fixed exposure (minimum)
    camera_params.append(f"exposuretimerange=\"{exposure} {exposure}\"")
    camera_params.append("aelock=true")  # Lock auto exposure
    
    # Configure gain
    if not auto_gain:
        camera_params.append("gainrange=\"1.0 1.0\"")  # Fixed gain at 1.0
        
    camera_param_str = " ".join(camera_params)

    # Construct the camera parameters
    camera_param_str = f"camera_id={camera_id} exposure={exposure} auto_gain=1"
    pipeline_elements = [
        f"nvarguscamerasrc {camera_param_str}",
        f"video/x-raw(memory:NVMM), width={resolution[0]}, height={resolution[1]}, framerate={framerate}/1",
        "nvvidconv",
        "video/x-raw, format=BGRx",
        "videoconvert",
        "autovideosink"
    ]

    if preview:
        pipeline_elements.append(
            f"textoverlay text=\"Camera Preview: Res:[{resolution[0]},{resolution[1]}] Exp={exposure}ns Bitrate={bitrate}\" "
            "valignment=top halignment=left font-desc=\"Sans, 24\""
        )

    # Create the complete pipeline string
    pipeline_str = " ! ".join(pipeline_elements)