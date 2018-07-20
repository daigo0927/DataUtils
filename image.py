from torch.utils.data import Dataset
from pathlib import Path
from itertools import islice
from fabric import colors
import numpy as np
import imageio
import torch
import random
import cv2
import warnings
from functools import partial
from abc import abstractmethod, ABCMeta

from . import utils


class BaseDataset(Dataset, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self): pass
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = imageio.imread(img_path)
        label = int(label)

        if image.ndim == 2: # gray -> rgb
            image = np.stack([image]*3, axis = 2)

        if self.crop_shape is not None:
            cropper = utils.StaticRandomCrop(image.shape[:2], self.crop_shape) if self.cropper == 'random'\
              else utils.StaticCenterCrop(image.shape[:2], self.crop_shape)
            image = cropper(image)

        if self.resize_shape is not None:
            image = cv2.resize(image, tuple(self.resize_shape[::-1]))

        elif self.resize_scale is not None:
            image = cv2.resize(image, dsize = (0,0), fx = self.resize_scale, fy = self.resize_scale)

        if self.one_hot:
            label = utils.one_hot(label, self.num_classes)

        if self.use_label:
            return np.array(image), label
        else:
            return np.array(image)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val + '.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_path, label = i.split(',')
                label = label.strip()
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

    @abstractmethod
    def get_classes(self): pass


class Food101(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, cropper = 'center',
                 crop_shape = None, resize_shape = None, resize_scale = None,
                 one_hot = True, use_label = False):
        super().__init__()
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale
        self.one_hot = one_hot
        self.use_label = use_label

        self.dataset_dir = dataset_dir
        self.train_or_val = train_or_val
        self.get_classes()
        p = Path(dataset_dir) / (train_or_val + '.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()

        warnings.filterwarnings('ignore', category = UserWarning)

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        assert self.train_or_val in ['train', 'test'], 'Argument should be either [train, test] in food-101 dataset.'
        p_set = p / 'meta' / (self.train_or_val+'.txt')
        p_img = p / 'images'
        self.samples = []

        with open(p_set, 'r') as f:
            for i in f.readlines():
                i = i.strip()
                class_ = i.split('/')[0]
                sample = (str(p_img/(i+'.jpg')), str(self.classes.index(class_)))
                self.samples.append(sample)

        if self.train_or_val is 'train':
            with open(p/'train.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in self.samples))
        else:
            with open(p/'val.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in self.samples))
        print(colors.green('**Food-101 dataset originally has train/test.txt in Food-101/meta,'+\
                           ' please don\'t mistake for the generated files**'))
                
    def get_classes(self):
        p = Path(self.dataset_dir)
        p_class = p / 'meta/classes.txt'
        with open(p_class, 'r') as f:
            self.classes = f.read().split('\n')[:-1]
        self.num_classes = len(self.classes) # 101 classes
            
