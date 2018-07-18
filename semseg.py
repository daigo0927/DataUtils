from torch.utils.data import Dataset
from pathlib import Path
from functools import partial
from itertools import islice
from tqdm import tqdm
from fabric import colors
import numpy as np
import imageio
import torch
import random
import cv2
import warnings

from abc import abstractmethod, ABCMeta

from . import utils


class BaseDataset(Dataset, metaclass = ABCMeta):
    @abstractmethod
    def __init__(self): pass
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        image, label = map(imageio.imread, (img_path, lbl_path))
        image, label = map(np.array, (image, label))
        label = self.encode_segmap(label)

        if image.ndim == 2: # gray -> rgb
            image = np.stack([image]*3, axis = 2)

        if self.crop_shape is not None:
            cropper = utils.StaticRandomCrop(image.shape[:2], self.crop_shape) if self.cropper == 'random'\
              else utils.StaticCenterCrop(image.shape[:2], self.crop_shape)
            image, label = map(cropper, (image, label))

        if self.resize_shape is not None:
            resizer = partial(cv2.resize, dsize = (0, 0), dst = self.resize_shape)
            image = resizer(image)
            label = resizer(label, interpolation = cv2.INTER_NEAREST)

        elif self.resize_scale is not None:
            resizer = partial(cv2.resize, dsize = (0, 0), fx = self.resize_scale, fy = self.resize_scale)
            image = resizer(image)
            label = resizer(label, interpolation = cv2.INTER_NEAREST)

        return image, label

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val + '.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_path, lbl_path = i.split(',')
                lbl_path = lbl_path.strip()
                self.samples.append((img_path, lbl_path))

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
    def decode_segmap(self, segmap):
        pass

    @abstractmethod
    def encode_segmap(self, segmap):
        pass

    
class CityScapes(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, cropper = 'random',
                 crop_shape = None, resize_shape = None, resize_scale = None):
        super().__init__()
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale

        self.set_property()
        self.dataset_dir = dataset_dir
        self.train_or_val = train_or_val
        p = Path(dataset_dir) / (train_or_val + '.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        assert self.train_or_val in ['train', 'val'], 'Argument should be either [train, val]'
        p_img = p / 'leftImg8bit' / self.train_or_val
        p_lbl = p / 'gtFine' / self.train_or_val
        self.samples = []

        image_paths = sorted(map(str, p_img.glob('**/*.png')))
        label_paths = sorted(map(str, p_lbl.glob('**/*labelIds.png')))

        for img_p, lbl_p in tqdm(zip(image_paths, label_paths)):
            self.samples.append((img_p, lbl_p))
        
        with open(p / f'{self.train_or_val}.txt', 'w') as f:
            f.writelines((','.join(i) + '\n' for i in self.samples))
        print(colors.green(f'\rImage and label path file is saved in {p}.'))

    def set_property(self):
        self.n_classes = 19
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        self.colors = [#[  0,   0,   0],
                       [128,  64, 128],
                       [244,  35, 232],
                       [ 70,  70,  70],
                       [102, 102, 156],
                       [190, 153, 153],
                       [153, 153, 153],
                       [250, 170,  30],
                       [220, 220,   0],
                       [107, 142,  35],
                       [152, 251, 152],
                       [  0, 130, 180],
                       [220,  20,  60],
                       [255,   0,   0],
                       [  0,   0, 142],
                       [  0,   0,  70],
                       [  0,  60, 100],
                       [  0,  80, 100],
                       [  0,   0, 230],
                       [119,  11,  32]]

        self.label_colors = dict(zip(range(19), self.colors))
        

    def decode_segmap(self, segmap):
        r = segmap.copy()
        g = segmap.copy()
        b = segmap.copy()
        for l in range(0, self.n_classes):
            r[segmap == l] = self.label_colors[l][0]
            g[segmap == l] = self.label_colors[l][1]
            b[segmap == l] = self.label_colors[l][2]

        rgb = np.zeros((segmap.shape[0], segmap.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


class SYNTHIA(BaseDataset):
    
