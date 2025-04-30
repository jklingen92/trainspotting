

def build_pipeline(camera_id, resolution, framerate, exposure, bitrate, preview=False):
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

    # Construct the camera parameters
    pipeline_elements = [
        f"nvarguscamerasrc camera_id={camera_id} sensor-mode=1 exposure={exposure}",
        f"video/x-raw(memory:NVMM), width={resolution[0]}, height={resolution[1]}, framerate={framerate}/1",
        "nvvidconv",
        "video/x-raw, format=BGRx",
        "videoconvert",
        "video/x-raw, format=BGRx",
        "appsink max-buffers=1 drop=true",
    ]

    if preview:
        pipeline_elements.append(
            f"textoverlay text=\"Camera Preview: Res:[{resolution[0]},{resolution[1]}] Exp={exposure}ns Bitrate={bitrate}\" "
            "valignment=top halignment=left font-desc=\"Sans, 24\""
        )

    # Create the complete pipeline string
    return " ! ".join(pipeline_elements)