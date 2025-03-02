# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
from PIL import Image


class VOCDataset(BaseDataSet):
    """
        my VOC-like dataset
    """

    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')

        file_list = os.path.join(self.root, self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class MyVOC(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False,
                 shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):
        # update at 2021.01.29
        self.MEAN = [0.4935838, 0.48873937, 0.45739236]
        self.STD = [0.22273207, 0.22567303, 0.22986929]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = VOCDataset(**kwargs)

        super(MyVOC, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
