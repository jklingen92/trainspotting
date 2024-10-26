import cv2
import os
import numpy as np

from json import load
from django.conf import settings
from django.core.management.base import BaseCommand

from detection.utils import ImageInterface, Video, Clip, milli2timestamp, FFMPEG_BASE


class Command(BaseCommand):
    help = "Isolates and clips motion events from a video file"

    def add_arguments(self, parser):
        parser.add_argument('videos', nargs="+")
        parser.add_argument('-d', '--dest')
        parser.add_argument('-u', '--upper', default=5, type=float)
        parser.add_argument('-l', '--lower', default=1, type=float)
        parser.add_argument('-m', '--minlength', default=3, type=int)
        parser.add_argument('-b', '--buffer', default=5, type=int)
        parser.add_argument('-f', '--fake', default=False, action='store_true')
        parser.add_argument('--location')
        parser.add_argument('--nomerge', action='store_true', default=False)


    def handle(self, *args, **options):
        dest = self.initialize_dest(options)
        videos, (x1, y1), (x2, y2) = self.initialize_videos(options["videos"])
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        unfinished = None 
        n_videos = len(options['videos'])  
        while True:
            try:
                gap, video = next(videos)
            except StopIteration:
                if unfinished is not None:
                    self.stdout.write(f"  Unable to finish clip: {unfinished.path}")
                self.stdout.write(f"Processed {n_videos}. Clips deposited in {dest}")
                break
            
            if gap:
                if unfinished:
                    self.stdout.write(f"  Found gap, unable to finish clip: {unfinished.path}")
                    unfinished = None
                else:
                    self.stdout.write(f"  Found gap, no clips compromised.")
            
            self.stdout.write(f"Processing video {i + 1} of {n_videos}: {video.path}")
            tail_frames = video.fps * options['buffer']
            minlength = options['buffer'] + options['minlength']
            clips = []

            if unfinished is None:
                frame0 = None
                clip = None
            else:
                self.stdout.write(f"  Found unfinished clip, seeking matching frame...")
                frame0, clip = self.seek_matching_frame(video, unfinished.frame)
                clip.merge_path = unfinished.path
                unfinished = None
            while True:
                success, frame1 = video.cap.read()

                # Handle end of video
                if not success:
                    if clip is not None:
                        unfinished = self.Unfinished(frame0)
                        clips.append(clip)
                    break
                
                # Handle first image
                if frame0 is None:
                    frame0 = self.process_frame(frame1)
                    continue

                frame1 = self.process_frame(frame1)
                delta = self.compare_frames(frame0, frame1)
                
                # Output
                if clip is not None:
                    status_str = f"Capturing: {milli2timestamp(video.pos_milli - clip.start)} | Stills: {clip.stills:4d}"
                else:
                    status_str = "-" * 30
                self.stdout.write(f"Progress: {milli2timestamp(video.pos_milli):8} | {delta:06.3f} | {status_str}", ending="\r")
                
                # Handle movement between frames
                if delta > options["upper"]:
                    if clip is not None:
                        clip.stills = 0
                    else:
                        clip = Clip(video.pos_milli)
                        clip.buff_start(options['buffer'])
                # Handle a still image
                elif delta < options["lower"] and clip is not None:
                    clip.stills += 1
                    if clip.stills >= tail_frames:
                        clip.stills = 0
                        clip.end = video.pos_milli
                        if (unfinished is not None and unfinished.defer) or clip.duration >= minlength * 1000:
                            clips.append(clip)
                        clip = None

                frame0 = frame1
            
            # Use FFMPEG to write the clips to a file
            self.stdout.write()
            i = 0
            for clip in clips:
                outfile = os.path.join(dest, clip.dest(video))

                self.stdout.write(f"  Clipping {i + 1} of {len(clips)}: {milli2timestamp(clip.start)} - {milli2timestamp(clip.end) if clip.end else 'End of video'}")
                if not options["fake"]:
                    end_str = ""
                    if clip.end is not None:
                        end_str = f"-to '{clip.end}ms'"
                    cmd = f"{FFMPEG_BASE} -ss '{clip.start}ms' {end_str} -i {video.path} -enc_time_base:v 1:24 {outfile}"
                    self.stdout.write(cmd)
                    os.system(cmd)
                    
                    if clip.merge_path and not options['nomerge']:
                        # This block merges the first clip of the current video with the previous unfinished clip
                        mergefile = os.path.join(dest, 'merge.mp4')
                        os.system(f'echo "file {clip.merge_path}" >> merge.txt')
                        os.system(f'echo "file {outfile}" >> merge.txt')
                        os.system(f"{FFMPEG_BASE} -f concat -safe 0 -i merge.txt -c copy {mergefile}")
                        os.system(f"mv {mergefile} {clip.merge_path}")
                        os.system(f"rm -f merge.txt {outfile}")
                
            # Check the last clip to see if it was truncated
            if unfinished is not None and unfinished.defer:
                unfinished.path = outfile
                unfinished.defer = False

    def initialize_dest(self, options):
        """Determine destination folder and create if necessary."""
        if options['dest'] is None:
            dest = os.path.join(f"{settings.BASE_DIR}", f"clips")
            if options["location"] is not None:
                dest = os.path.join(dest, options["location"])
        else:
            dest = options["dest"]
        
        if not os.path.exists(dest):
            os.makedirs(dest)

        return dest

    def initialize_videos(self, video_paths):
        """Sort videos by timestamp, check for gaps, and get a bounding box to watch for motion."""
        self.stdout.write("Processing videos initially...")
        video_paths.sort(key=lambda p: os.path.getctime(p))

        video0 = Video(video_paths[0])
        success, img = video0.cap.read()
        if success:
            interface = ImageInterface(img)
            (x1, y1), (x2, y2) = interface.get_bounding_box()
        else:
            raise Exception(f"{video0} has no frames!")
        video0.release()

        def video_generator(video_paths):
            last_video_end = None
            for video_path in video_paths:
                video = Video(video_path)
                gap = last_video_end is None or last_video_end > video.start
                yield gap, video
                last_video_end = video.end

        return video_generator(video_paths), (x1, y1), (x2, y2)

    def process_frame(self, frame):
        frame = frame[self.y1:self.y2, self.x1:self.x2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        return frame
    
    def compare_frames(self, frame1, frame2):
        diff = cv2.absdiff(frame1, frame2)
        score = np.mean(diff)
        return score

    class Unfinished:
        def __init__(self, frame) -> None:
            self.defer = True   # Kind of hacky way to defer processing
            self.frame = frame
            self.path = None

    def seek_matching_frame(self, video, frame):
        """Find the first frame that matches the given frame and set up a clip at that point."""
        # Get the video offset
        os.system(f'ffprobe -show_entries stream=codec_type,start_time -v 0 -of json {video.path} >> offsets.json')
        with open("offsets.json") as f:
            offsets = load(f)
        os.system(f"rm -f offsets.json")
        
        video_offset = 0
        smallest_offset = np.inf
        for stream in offsets["streams"]:
            if stream["codec_type"] == "video":
                video_offset = float(stream["start_time"])
            smallest_offset = min(float(stream["start_time"]), smallest_offset)
        
        offset = (video_offset - smallest_offset) * 1000

        # Seek the frame
        while True:
            success, img = video.cap.read()

            if not success:
                raise Exception("Did not find matching frame.")

            diff = self.compare_frames(frame, self.process_frame(img))
            if diff == 0.0:
                _, frame = video.cap.read()
                clip = Clip(video.pos_milli + offset)
                break
        
        return self.process_frame(frame), clip

