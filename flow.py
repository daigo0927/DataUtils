from torch.utils.data import Dataset
from pathlib import Path
from itertools import islice
import numpy as np
import imageio
import torch
import random
import cv2
from functools import partial
from abc import abstractmethod, ABCMeta

from . import utils
from .flow_utils import load_flow


class BaseDataset(Dataset, metaclass = ABCMeta):
    def __init__(self, dataset_dir, train_or_val, color = 'rgb',
                 cropper = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        self.dataset_dir = dataset_dir
        assert train_or_val in ['train', 'val'], 'Argument should be either of [train, val]'
        self.train_or_val = train_or_val
        self.color = color
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = crop_shape
        self.resize_scale = resize_scale
        self.flow_reader = load_flow

        p = Path(dataset_dir) / (train_or_val + '.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img1_path, img2_path, flow_path = self.samples[idx]
        img1, img2 = map(imageio.imread, (img1_path, img2_path))
        flow = self.flow_reader(flow_path)

        if self.color == 'gray':
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]

        images = [img1, img2]
        if self.crop_shape is not None:
            cropper = utils.StaticRandomCrop(img1.shape[:2], self.crop_shape) if self.cropper == 'random'\
              else utils.StaticCenterCrop(img1.shape[:2], self.crop_shape)
            images = list(map(cropper, images))
            flow = cropper(flow)
        if self.resize_shape is not None:
            resizer = partial(cv2.resize, dsize = tuple(self.resize_shape[::-1])) # put as (x, y) order
            images = list(map(resizer, images))
            flow = resizer(flow)
        elif self.resize_scale is not None:
            resizer = partial(cv2.resize, dsize = (0,0), fx = self.resize_scale, fy = self.resize_scale)
            images = list(map(resizer, images))
            flow = resizer(flow)

        images = np.array(images)
        return images, flow
    
    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val + '.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img1, img2, flow = i.split(',')
                flow = flow.strip()
                self.samples.append((img1, img2, flow))

    @abstractmethod
    def has_no_txt(self): ...
    
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


# FlyingChairs
# ============================================================
class FlyingChairs(BaseDataset):
    def __init__(self, dataset_dir, train_or_val = 'train', color = 'rgb',
                 cropper = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, color, cropper, crop_shape, resize_shape, resize_scale)

    def has_no_txt(self):
        p = Path(self.dataset_dir) / 'data'
        imgs = sorted(p.glob('*.ppm'))
        samples = [(str(i[0]), str(i[1]), str(i[0]).replace('img1', 'flow').replace('.ppm', '.flo'))\
                   for i in zip(imgs[::2], imgs[1::2])]
        self.split(samples)


# FlyingThings
# ============================================================
class FlyingThings(BaseDataset):
    def __init__(self): ...


# Sintel
# ============================================================
class Sintel(BaseDataset):
    def __init__(self, dataset_dir, train_or_val, mode = 'final', color = 'rgb',
                 cropper = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, color, cropper, crop_shape, resize_shape, resize_scale)
        self.mode = mode
    
    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_img = p / 'training' / self.mode
        p_flow = p / 'training/flow'
        samples = []

        collections_of_scenes = sorted(map(str, p_img.glob('**/*.png')))
        from itertools import groupby
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('/')[-2])]

        samples = [(*i, i[0].replace(self.mode, 'flow').replace('.png', '.flo'))\
                   for collection in collections for i in utils.window(collection, 2)]
        self.split(samples)

class SintelFinal(Sintel):
    def __init__(self, dataset_dir, train_or_val, color = 'rgb',
                 cropper = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, 'final', color,
                         cropper, crop_shape, resize_shape, resize_scale)

class SintelClean(Sintel):
    def __init__(self, dataset_dir, train_or_val, color = 'rgb',
                 cropper = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, 'clean', color,
                         cropper, crop_shape, resize_shape, resize_scale)

# KITTI
# ============================================================
class KITTI(BaseDataset):
    def __init__(self, dataset_dir, train_or_val = 'train', color = 'rgb',
                 cropper = 'random', crop_shape = (320, 448),
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, color, cropper, crop_shape, resize_shape, resize_scale)
        self.flow_reader = self._readflow

    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_img = p / 'training/image_2'
        p_flow = p / 'training/flow_occ'
        samples = []
        for i in range(200):
            p_i0 = p_img / f'{str(i).zfill(6)}_10.png'
            p_i1 = p_img / f'{str(i).zfill(6)}_11.png'
            p_f = p_flow / f'{str(i).zfill(6)}_10.png'
            samples.append(tuple(map(str, (p_i0, p_i1, p_f))))
        self.split(samples)

    def _readflow(self, uri):
        # flow_origin = imageio.imread(uri, format = 'PNG-FI')
        flow_origin = cv2.imread(uri, cv2.IMREAD_UNCHANGED)
        flow = flow_origin[:, :, 2:0:-1].astype(np.float32)
        invalid = (flow_origin[:, :, 0] == 0)
        flow = (flow - 2**15)/64
        flow[np.abs(flow) < 1e-10] = 1e-10
        flow[invalid] = 0
        return flow
