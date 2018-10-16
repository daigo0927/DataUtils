from torch.utils.data import Dataset
from pathlib import Path
from itertools import groupby
import numpy as np
import imageio
import torch
import random
import cv2
from functools import partial
from abc import abstractmethod, ABCMeta

from . import utils


def load_flow(uri):
    """
    Function to load optical flow data
    Args: str uri: target flow path
    Returns: np.ndarray: extracted optical flow
    """
    with open(uri, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None

def resize_flow(flow, resize_shape):
    if flow.ndim != 3:
        raise ValueError(f'Flow dimension should be 3, but found {flow.ndim} dimension')
    h, w = flow.shape[:2]
    th, tw = resize_shape # target size
    scale = np.array([tw/w, th/h]).reshape((1, 1, 2))
    flow = cv2.resize(flow, dsize = (tw, th))*scale
    flow = np.float32(flow)
    return flow

def rescale_flow(flow, resize_scale):
    if flow.ndim != 3:
        raise ValueError(f'Flow dimension should be 3, but found {flow.ndim} dimension')
    h, w = flow.shape[:2]
    th, tw = int(h*resize_scale[0]), int(w*resize_scale[1])
    scale = np.array(resize_scale).reshape((1, 1, 2))
    flow = cv2.resize(flow, dsize = (tw, th))*scale
    flow = np.float32(flow)
    return flow


class BaseDataset(Dataset, metaclass = ABCMeta):
    """ Abstract class to flexibly utilize torch.data pipeline """
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        """ 
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - origin_size tuple<int>: original size of target images
        - crop_type str: crop type, 'random', 'center', or None
        - crop_shape tuple<int>: crop shape
        - resize_shape tuple<int>: resize shape
        - resize_scale tuple<int>: resize scale (<= 1)
        """
        self.dataset_dir = dataset_dir
        assert train_or_val in ['train', 'val'], 'Argument should be either of [train, val]'
        self.train_or_val = train_or_val

        self.image_size = utils.get_size(origin_size, crop_shape, resize_shape, resize_scale)
        self.crop_type = crop_type
        self.crop_shape = crop_shape
        self.resize_shape = crop_shape
        self.resize_scale = resize_scale

        p = Path(dataset_dir) / (train_or_val + '.txt')
        if p.exists(): self.has_txt()
        else: self.has_no_txt()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img0_path, img1_path, flow_path = self.samples[idx]
        image_0, image_1 = map(imageio.imread, (img0_path, img1_path))
        flow = self.load_flow(flow_path)
        
        if self.crop_shape is not None:
            cropper = utils.StaticRandomCrop(image_0.shape[:2], self.crop_shape) if self.crop_type == 'random'\
              else utils.StaticCenterCrop(image_0.shape[:2], self.crop_shape)
            image_0, image_1, flow = map(cropper, [image_0, image_1, flow])
            
        if self.resize_shape is not None:
            image_0 = cv2.resize(image_0, dsize = tuple(self.resize_shape[::-1]))
            image_1 = cv2.resize(image_1, dsize = tuple(self.resize_shape[::-1]))
            flow = resize_flow(flow, self.resize_shape)
            
        if self.resize_scale is not None:
            sx, sy = self.resize_scale
            image_0 = cv2.resize(image_0, dsize = (0, 0), fx = sx, fy = sy)
            image_1 = cv2.resize(image_1, dsize = (0, 0), fx = sx, fy = sy)
            flow = rescale_flow(flow, self.resize_scale)

        images = np.stack([image_0, image_1], axis = 0)
        return images, flow
    
    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val + '.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_0_path, img_1_path, flow_path = i.split(',')
                flow_path = flow_path.strip()
                self.samples.append((img_0_path, img_1_path, flow_path))

    @abstractmethod
    def has_no_txt(self): ...
    
    def split(self, samples):
        p = Path(self.dataset_dir)
        val_ratio = 0.1
        random.shuffle(samples)
        idx = int(len(samples) * (1 - val_ratio))
        train_samples = samples[:idx]
        val_samples = samples[idx:]

        with open(p / 'train.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in train_samples))
        with open(p / 'val.txt', 'w') as f: f.writelines((','.join(i) + '\n' for i in val_samples))

        self.samples = train_samples if self.train_or_val == 'train' else val_samples

    def load_flow(self, flow_path):
        return load_flow(flow_path)


# FlyingChairs
# ============================================================
class FlyingChairs(BaseDataset):
    """ FlyingChairs dataset pipeline """
    def __init__(self, dataset_dir, train_or_val = 'train', origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, origin_size,
                         crop_type, crop_shape, resize_shape, resize_scale)

    def has_no_txt(self):
        p = Path(self.dataset_dir) / 'data'
        imgs = sorted(p.glob('*.ppm'))
        samples = [(str(i[0]), str(i[1]), str(i[0]).replace('img1', 'flow').replace('.ppm', '.flo'))\
                   for i in zip(imgs[::2], imgs[1::2])]
        self.split(samples)


# FlyingThings
# ============================================================
class FlyingThings3D(BaseDataset):
    def __init__(self, dataset_dir, train_or_val = 'train', origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, origin_size,
                         crop_type, crop_shape, resize_shape, resize_scale)

    def has_no_txt(self):
        # TODO
        pass

    
# Sintel
# ============================================================
class Sintel(BaseDataset):
    """ MPI-Sintel-complete dataset pipeline """
    def __init__(self, dataset_dir, train_or_val, mode = 'clean',
                 origin_size = None, crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        self.mode = mode
        super().__init__(dataset_dir, train_or_val, origin_size,
                         crop_type, crop_shape, resize_shape, resize_scale)
    
    def has_no_txt(self):
        p = Path(self.dataset_dir)
        p_img = p / 'training' / self.mode
        p_flow = p / 'training/flow'
    
        collections_of_scenes = sorted(map(str, p_img.glob('**/*.png')))
        collections = [list(g) for k, g in groupby(collections_of_scenes, lambda x: x.split('/')[-2])]
        samples = [(*i, i[0].replace(self.mode, 'flow').replace('.png', '.flo'))\
                   for collection in collections for i in utils.window(collection, 2)]
        self.split(samples)

class SintelClean(Sintel):
    """ MPI-Sintel-complete dataset (clean path) pipeline """
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, 'clean', origin_size,
                         crop_type, crop_shape, resize_shape, resize_scale)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val+'.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_0_path, img_1_path, flow_path = i.split(',')
                img_0_path, img_1_path = map(lambda p: p.replace('final', 'clean'),
                                             (img_0_path, img_1_path))
                flow_path = flow_path.strip()
                self.samples.append((img_0_path, img_1_path, flow_path))

class SintelFinal(Sintel):
    """ MPI-Sintel-complete dataset (clean path) pipeline """
    def __init__(self, dataset_dir, train_or_val, origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, 'final', origin_size,
                         crop_type, crop_shape, resize_shape, resize_scale)

    def has_txt(self):
        p = Path(self.dataset_dir) / (self.train_or_val+'.txt')
        self.samples = []
        with open(p, 'r') as f:
            for i in f.readlines():
                img_0_path, img_1_path, flow_path = i.split(',')
                img_0_path, img_1_path = map(lambda p: p.replace('clean', 'final'),
                                             (img_0_path, img_1_path))
                flow_path = flow_path.strip()
                self.samples.append((img_0_path, img_1_path, flow_path))

                
# KITTI
# ============================================================
class KITTI(BaseDataset):
    """ KITTI 2015 dataset """
    def __init__(self, dataset_dir, train_or_val = 'train', origin_size = None,
                 crop_type = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, train_or_val, origin_size,
                         crop_type, crop_shape, resize_shape, resize_scale)

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

    def load_flow(self, uri):
        # flow_origin = imageio.imread(uri, format = 'PNG-FI')
        flow_origin = cv2.imread(uri, cv2.IMREAD_UNCHANGED)
        flow = flow_origin[:, :, 2:0:-1].astype(np.float32)
        invalid = (flow_origin[:, :, 0] == 0)
        flow = (flow - 2**15)/64
        flow[np.abs(flow) < 1e-10] = 1e-10
        flow[invalid] = 0
        return flow

    
def get_dataset(dataset_name):
    """
    Get specified dataset 
    Args: dataset_name str: target dataset name
    Returns: dataset tf.data.Dataset: target dataset class

    Available dataset pipelines
    - FlyingChairs
    - Sintel, SintelClean, SintelFinal
    - KITTI
    """
    datasets = {"FlyingChairs":FlyingChairs,
                "Sintel":Sintel,
                "SintelClean":SintelClean,
                "SintelFinal":SintelFinal,
                "KITTI":KITTI}
    return datasets[dataset_name]
