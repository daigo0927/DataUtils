from torch.utils.data import Dataset
from pathlib import Path
from itertools import islice
from fablic import colors
import numpy as np
import imageio
import torch
import random
import cv2
from functools import partial
from abc import abstractmethod, ABCMeta

import utils


class BaseDataset(Dataset, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self): pass
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = imageio.imread(img_path)

        if self.crop_shape is not None:
            cropper = utils.StaticRandomCrop(image.shape[:2], self.crop_shape) if self.cropper == 'random'\
              else utils.StaticCenterCrop(image.shape[:2], self.crop_shape)
            image = cropper(image)

        if self.resize_shape is not None:
            image = cv2.resize(image, dsize = (0,0), dst = self.resize_shape)

        elif self.resize_scale is not None:
            image = cv2.resize(image, dsize = (0,0), fx = self.resize_scale, fy = self.resize_scale)

        if self.one_hot:
            label = one_hot(label)

        return np.array(image), label

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val + '.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_path, label = i.split(',')
                label = int(label.strip())
                self.samples.append((img_path, label))

    @abstractmethod
    def has_no_txt(self): pass

    def split(self, samples):
        p = Path(self.dataset_dir)
        test_ratio = 0.1
        random.shuffle(samples)
        idx = int(len(samples) * (1 - test_ratio))
        train_samples = samples[:idx]
        val_samples = samples[idx:]

        with open(p / 'train.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
        with open(p / 'val.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in val_samples))

        self.samples = train_samples if self.train_or_val == 'train' else val_samples


class Food101(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, cropper = 'center',
                 crop_shape = None, resize_shape = None, resize_scale = None):
        super().__init__()
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale

        self.dataset_dir = dataset_dir
        self.train_or_val = train_or_val
        p = Path(dataset_dir) / (train_or_val + '.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()

    def has_no_txt(self):
        
