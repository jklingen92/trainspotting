from django.conf import settings
from fastai.vision.all import *



class TrainSpotter:

    def __init__(self) -> None:
        self.dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=[Resize(192, method='squish')]
        ).dataloaders(os.path.join(settings.BASE_DIR, "classification/training/train_or_not"), bs=32)
        
        self.learn = vision_learner(self.dls, resnet18, metrics=error_rate)
        self.learn.fine_tune(3)