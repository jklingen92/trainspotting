

import os

from django.conf import settings
from detection.detector import Detector, ExclusionDetector
from detection.models import Detection, VideoBatch, VideoHandler
from detection.utils import image_from_array


def import_videos(video_paths: list [str], camera: str, logger=None):

    video_batch = VideoBatch.objects.create(camera=camera)
    num_videos = len(video_paths)
    
    staging_location = os.path.join(settings.MEDIA_ROOT, 'raw', camera.name)
    if not os.path.exists(staging_location):
        if logger:
            logger.info("Creating video repository: {staging_location}")
        os.makedirs(staging_location)
    
    if logger:
        logger.info(f"Importing {num_videos} videos to {staging_location}:")
        
    try:
        for i, video_path in enumerate(video_paths):
            filename = video_path.split('/')[-1]

            if logger:
                logger.info(f"  Importing {filename} ({i + 1} of {num_videos})")

            os.system(f"rsync {video_path} {staging_location}/")
            task = VideoHandler.objects.create(batch=video_batch, file=os.path.join('raw', camera.name, filename))
            task.init()
    
    except Exception as e:
        if logger:
            logger.error(e)
        else:
            raise e
    
    finally:
        if logger:
            logger.info(f"Successfully created {video_batch.processing_tasks.count()} processing_tasks")

    return video_batch


def detect_clips(video_handlers, view: str ="", logger=None):

        # Assert that there is only one camera among the video_handlers
        assert video_handlers.values("camera").distinct().count() == 1
        camera = video_handlers.first().camera

        first_handler = video_handlers.first()
        first_frame = first_handler.read()
        first_handler.release()

        sample_name = f"{first_handler.video.filename}_sample.png"
        detection, _ = Detection.objects.get_or_create(
            camera=camera,
            view=view,
            defaults={
                "sample": image_from_array(sample_name, first_frame.image)
            }
        )
        
        detection.detect_area = detection.get_bounding_box("Select a detection area or leave blank for the full area")
        detection.exclude_area = detection.get_bounding_box("Select an exclusion area or leave blank for no exclusion")
        detection.save()

        if detection.exclude_area is None:
            detector = Detector(detection, video_handlers, logger=logger)
        else:
            detector = ExclusionDetector(detection, logger=logger)

        try:
            detector.detect_loop()
            if logger:
                logger.info(f"Created {detection.clips.count()} clips")

        except Exception as e:
            if logger:
                logger.error(e)
        else:
            raise e
        
        return detection


def extract_clips(detection: Detection, logger=None):

    clips = detection.clips.all()
    
    if logger:
        logger.info(f"Extracting {clips.count()} clips to {detection.clip_destination}...")
    
    for clip in clips:
        if logger:
            logger.info(f"  Extracting {clip.outfile}...")
        clip.extract()
    
    if logger:
        logger.info(f"Extracted {clips.count()} clips to {detection.clip_destination}!")

