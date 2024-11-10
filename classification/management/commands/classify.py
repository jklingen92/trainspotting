import os
import cv2
from django.core.management.base import BaseCommand
from random import randint
from classification.utils import TrainSpotter
from trainspotting.utils import Video, VideosCommand


class Command(BaseCommand):
    help = "Classifies clips into train or no train"

    def add_arguments(self, parser):
        parser.add_argument('folder')

    def handle(self, *args, **options):
        videos = os.listdir(options['folder'])
        train_folder = os.path.join(options['folder'], 'train')
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        not_train_folder = os.path.join(options['folder'], 'not_train')
        if not os.path.exists(not_train_folder):
            os.makedirs(not_train_folder)
        t = TrainSpotter()
        
        for filename in videos:
            video_path = os.path.join(options['folder'], filename)
            video = Video(video_path)
            video.seek_frame(video.frame_count // 2)
            s, img = video.cap.read()
            is_train, _, prob = t.learn.predict(img)
            if is_train and prob[0] >= 0.8:
                
                os.rename(video_path, os.path.join(train_folder, filename))
            else:
                os.rename(video_path, os.path.join(not_train_folder, filename))